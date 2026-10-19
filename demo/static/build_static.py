"""Build the static (in-browser) demo: ONNX models, tokenizer files, question bank, web files.

usage: python -m demo.static.build_static --checkpoints <dir> --out <dir> [--scores <summary.json>] [--weights fp16|int8|fp32]

<dir> holds slim checkpoints (demo/build_space.py slim_checkpoint): <lang>_lm.pt (pretrained
V2) and the reasoning fine-tunes <lang>_<arch>_<mode>.pt, arch v1/v2 and mode direct/cot
(<lang>_reasoning.pt, as build_space names it, stands for v1_direct). Missing variants are
skipped. --scores is the eval_hard/summary.json of the lma-eval-hard kernel: each variant's
relation accuracy per hop depth goes into data/models.json for the page to show.
The output is the demo's only UI: index.html + app.js + lm.js run the models in the browser
with ONNX Runtime Web, so no GPU is needed. demo/build_space.py --site packages it with the
server (demo/app.py) for the Hugging Face Space and Railway; any static web host also works.

Each model is exported as one step of generation with a key/value cache:
  inputs   input_ids (1, T) int64, past_k / past_v (n_layer, 1, n_head, P, d_k) float32
  outputs  logits (1, vocab) for the last position, present_k / present_v (..., P + T, d_k)
P = 0 for the prompt, then T = 1 per new token. The wrappers below re-implement the
forward pass from the trained modules without Python branches on P, so one graph serves
both cases. Weights are stored as fp16 (the slim checkpoints already are, so nothing is lost)
and cast to fp32 when the session starts: ~64 MB per model (the tied embedding is stored twice). int8 dynamic quantization
(~32 MB) changes the reasoning models' answers: they must copy unseen names exactly, and
int8 garbles them. check_export compares the final file with the PyTorch model.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from demo.build_space import build_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LANGS = ("hindi", "assamese")
WEB_FILES = ("index.html", "app.js", "lm.js", "translit.js")
QUESTION_HOPS = (2, 3, 4, 5, 6, 8, 10)
QUESTIONS_PER_HOP = 40
VARIANTS = ("v1_direct", "v2_direct", "v1_cot", "v2_cot")
# hops -> eval_hard split; the main test split (mostly 2-3 hops) stands in for 2
SCORE_SPLITS = {2: "test", 3: "test_depth3", 4: "test_depth4", 5: "test_depth5",
                6: "test_depth6", 8: "test_depth8", 10: "test_depth10"}


class Step(nn.Module):
    """One cached generation step of a V1 (GPTLanguageModel) or V2 (GPTLanguageModelV2) model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.m = model
        self.v2 = hasattr(model, "norm_f")
        attn = model.blocks[0].attn
        self.n_head, self.d_k, self.d_model = attn.n_head, attn.d_k, attn.d_model

    def forward(self, input_ids, past_k, past_v):
        T, P = input_ids.shape[1], past_k.shape[3]
        pos = torch.arange(T) + P                      # absolute positions of the new tokens
        keys = torch.arange(P + T)
        masked = keys[None, :] > pos[:, None]          # (T, P + T): future keys
        m = self.m
        x = m.token_embedding(input_ids)
        if not self.v2:
            x = x + m.position_embedding(pos)
        new_k, new_v = [], []
        for layer, block in enumerate(m.blocks):
            attn = block.attn
            h = (block.norm_1 if self.v2 else block.ln_1)(x)
            q, k, v = attn.c_attn(h).split(self.d_model, dim=2)
            q, k, v = (t.view(1, T, self.n_head, self.d_k).transpose(1, 2) for t in (q, k, v))
            if self.v2:
                cos, sin = attn.rope_cos[pos][None, None], attn.rope_sin[pos][None, None]
                q = q * cos + _rotate_half(q) * sin
                k = k * cos + _rotate_half(k) * sin
            k = torch.cat((past_k[layer], k), dim=2)
            v = torch.cat((past_v[layer], v), dim=2)
            new_k.append(k)
            new_v.append(v)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
            att = torch.softmax(att.masked_fill(masked[None, None], float("-inf")), dim=-1)
            y = (att @ v).transpose(1, 2).reshape(1, T, self.d_model)
            x = x + attn.c_proj(y)
            x = x + block.mlp((block.norm_2 if self.v2 else block.ln_2)(x))
        x = (m.norm_f if self.v2 else m.ln_f)(x[:, -1:, :])
        return m.lm_head(x)[:, 0, :], torch.stack(new_k), torch.stack(new_v)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def load_slim(path: Path) -> nn.Module:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(saved["config"]["model"])
    model.load_state_dict({k: v.float() for k, v in saved["model_state_dict"].items()})
    return model.eval()


def empty_cache(model: nn.Module):
    attn = model.blocks[0].attn
    return torch.zeros(len(model.blocks), 1, attn.n_head, 0, attn.d_k)


def export(model: nn.Module, out: Path) -> None:
    step = Step(model).eval()
    attn = model.blocks[0].attn
    ids = torch.randint(0, model.config.vocab_size, (1, 2))
    past = torch.randn(len(model.blocks), 1, attn.n_head, 3, attn.d_k)  # P > 0: concat traced generally
    with torch.no_grad():
        torch.onnx.export(
            step, (ids, past, past.clone()), str(out), dynamo=False, opset_version=17,
            input_names=["input_ids", "past_k", "past_v"],
            output_names=["logits", "present_k", "present_v"],
            dynamic_axes={"input_ids": {1: "T"}, "past_k": {3: "P"}, "past_v": {3: "P"},
                          "present_k": {3: "S"}, "present_v": {3: "S"}},
        )


def check_export(model: nn.Module, onnx_path: Path, tokenizer, text: str, n_new: int = 24) -> dict:
    """Greedy generation through the ONNX graph (prompt, then one token per step) vs the
    PyTorch model's full forward; returns the max logit difference and token agreement."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ids = tokenizer.encode(text)
    k = v = empty_cache(model).numpy()
    seq, step_ids, max_diff = list(ids), ids, 0.0
    for _ in range(n_new):
        logits, k, v = sess.run(None, {"input_ids": np.array([step_ids], dtype=np.int64), "past_k": k, "past_v": v})
        with torch.no_grad():
            ref = model(torch.tensor([seq]))["logits"][0, -1].numpy()
        max_diff = max(max_diff, float(np.abs(logits[0] - ref).max()))
        nxt = int(logits[0].argmax())
        if nxt != int(ref.argmax()):
            return {"max_logit_diff": max_diff, "greedy_agrees": False}
        seq.append(nxt)
        step_ids = [nxt]
    return {"max_logit_diff": max_diff, "greedy_agrees": True}


def store_fp16(src: Path, dst: Path) -> None:
    """Float weights stored as fp16, each followed by a Cast back to fp32 (constant-folded when
    ONNX Runtime loads the model), so the file halves and the computation stays fp32."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(src))
    casts = []
    for init in m.graph.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= 1024:
            name = init.name
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), f"{name}__fp16"))
            casts.append(helper.make_node("Cast", [f"{name}__fp16"], [name], to=TensorProto.FLOAT))
    nodes = [*casts, *m.graph.node]
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def quantize(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul", "Gather"])


def tokenizer_json(lang: str) -> dict:
    """Vocabulary of the SentencePiece model, for lm.js (BPE encode/decode)."""
    from sentencepiece import sentencepiece_model_pb2 as pb

    proto = pb.ModelProto()
    proto.ParseFromString((ROOT / lang / "tokenizer" / f"{lang}.model").read_bytes())
    # type: 1 normal, 2 unknown, 3 control, 6 byte
    return {"pieces": [[p.piece, round(p.score, 6), p.type] for p in proto.pieces],
            "unk_id": proto.trainer_spec.unk_id, "eos_id": proto.trainer_spec.eos_id}


def question_bank(lang: str, seed: int = 2026) -> list[dict]:
    """Fresh test-style questions (test-pool names, every paradigm) for each hop count."""
    import importlib

    gen = importlib.import_module(f"{lang}.finetune.generate_reasoning")
    rng = random.Random(seed)
    bank = []
    for hops in QUESTION_HOPS:
        examples = gen.generate_examples(QUESTIONS_PER_HOP, gen.TEST_ENTITY_POOL, gen.EVAL_PARADIGMS, "test",
                                         seed=rng.randrange(10**9), n_entities=hops + 1, exact_hops=hops)
        bank += [{"hops": hops, "text": e.text, "query": list(e.query), "answer": e.answer_text,
                  "relation": e.answer_rel.split()[1], "paradigm": e.paradigm} for e in examples]
    return bank


def find_checkpoint(ckpt_dir: Path, lang: str, role: str) -> Path | None:
    for name in (role, "reasoning") if role == "v1_direct" else (role,):
        if (ckpt_dir / f"{lang}_{name}.pt").exists():
            return ckpt_dir / f"{lang}_{name}.pt"
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scores", help="eval_hard/summary.json of the lma-eval-hard kernel")
    parser.add_argument("--weights", choices=["fp16", "int8", "fp32"], default="fp16",
                        help="weight storage (default fp16; int8 is smaller but changes reasoning answers)")
    args = parser.parse_args(argv)

    import importlib

    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    scores = json.loads(Path(args.scores).read_text(encoding="utf-8")) if args.scores else {}
    report, catalog = {}, {}
    for lang in LANGS:
        tok = importlib.import_module(f"{lang}.tokenizer.tokenizer").Tokenizer(
            str(ROOT / lang / "tokenizer" / f"{lang}.model"))
        probe = question_bank(lang)[0]["text"]
        catalog[lang] = []
        for role in ("lm", *VARIANTS):
            path = find_checkpoint(Path(args.checkpoints), lang, role)
            if path is None:
                print(f"  {lang}_{role}: no checkpoint, skipped", flush=True)
                continue
            name = f"{lang}_{role}"
            model = load_slim(path)
            attn = model.blocks[0].attn
            meta = {"n_layer": len(model.blocks), "n_head": attn.n_head, "d_k": attn.d_k,
                    "block_size": model.config.block_size, "vocab_size": model.config.vocab_size,
                    "arch": "v2" if hasattr(model, "norm_f") else "v1"}
            (out / "models" / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
            fp32 = out / "models" / f"{name}.fp32.onnx"
            export(model, fp32)
            final = out / "models" / f"{name}.onnx"
            if args.weights == "fp32":
                fp32.replace(final)
            else:
                (store_fp16 if args.weights == "fp16" else quantize)(fp32, final)
                fp32.unlink()
            report[name] = check_export(model, final, tok, probe, n_new=64)
            report[name]["mb"] = round(final.stat().st_size / 1e6, 1)
            print(f"  {name}: {report[name]}", flush=True)
            if role != "lm":
                arch, mode = role.split("_")
                entry = {"id": role, "arch": arch, "mode": mode, "mb": report[name]["mb"]}
                scored = scores.get(f"{lang}_{arch}", {}).get(mode)
                if scored:
                    entry["job"] = scored["job"]
                    entry["accuracy"] = {h: round(scored["splits"][s]["accuracy_relation"], 3)
                                         for h, s in SCORE_SPLITS.items() if s in scored["splits"]}
                catalog[lang].append(entry)
        (out / "data" / f"{lang}_tokenizer.json").write_text(json.dumps(tokenizer_json(lang), ensure_ascii=False), encoding="utf-8")
        (out / "data" / f"{lang}_questions.json").write_text(json.dumps(question_bank(lang), ensure_ascii=False), encoding="utf-8")
    for name in WEB_FILES:
        shutil.copy2(HERE / name, out / name)
    (out / "data" / "models.json").write_text(json.dumps(catalog, indent=1), encoding="utf-8")
    (out / "data" / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
