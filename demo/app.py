"""Gradio demo of the Hindi and Assamese models: free text generation and relational reasoning.

Runs on CPU. Expects checkpoints/<lang>_lm.pt (pretrained V2, free generation) and
checkpoints/<lang>_reasoning.pt (V1 after Direct reasoning fine-tuning), as written by
demo/build_space.py. Each language's tokenizer, model code and reasoning generator are
imported from its own package (hindi.*, assamese.*).
"""
from __future__ import annotations

import importlib
import random
import tempfile
from pathlib import Path

import gradio as gr
import torch
import yaml

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
LANGS = {"Hindi": "hindi", "Assamese": "assamese"}
MAX_PROMPT_TOKENS = 384  # leaves room for generation in the 512-token context
EXAMPLES = {
    "hindi": "भारत की राजधानी नई दिल्ली है और",
    "assamese": "অসমৰ ৰাজধানী গুৱাহাটী আৰু",
}
# Same as the evaluation's nucleus setting (Phase 2), which gave the best chrF.
DEFAULT_SAMPLING = {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.15}
torch.set_grad_enabled(False)


def _load_model(lang: str, cfg: dict) -> torch.nn.Module:
    v2 = cfg.get("arch_version") == "v2" or "rope_theta" in cfg
    mod = importlib.import_module(f"{lang}.model.{'gpt_v2' if v2 else 'gpt'}")
    config_cls, model_cls = (mod.GPTConfigV2, mod.GPTLanguageModelV2) if v2 else (mod.GPTConfig, mod.GPTLanguageModel)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return model_cls(config_cls.from_yaml(path))


class LanguageModels:
    """Tokenizer, generator module and the two checkpoints of one language."""

    def __init__(self, lang: str):
        self.lang = lang
        self.tokenizer = importlib.import_module(f"{lang}.tokenizer.tokenizer").Tokenizer(
            str(HERE / lang / "tokenizer" / f"{lang}.model"))
        self.gen = importlib.import_module(f"{lang}.finetune.generate_reasoning")
        self.models = {}
        for role in ("lm", "reasoning"):
            path = CKPT_DIR / f"{lang}_{role}.pt"
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                model = _load_model(lang, saved["config"]["model"])
                model.load_state_dict({k: v.float() for k, v in saved["model_state_dict"].items()})
                self.models[role] = model.eval()

    def generate(self, role: str, text: str, max_new_tokens: int, **sampling) -> str:
        ids = self.tokenizer.encode(text)[-MAX_PROMPT_TOKENS:]
        if not ids:
            return ""
        out = self.models[role].generate(torch.tensor([ids]), max_new_tokens=max_new_tokens,
                                         eos_id=self.tokenizer.eos_id, use_cache=True, **sampling)
        new = out[0, len(ids):].tolist()
        if self.tokenizer.eos_id in new:
            new = new[: new.index(self.tokenizer.eos_id)]
        # Decode prompt and continuation together: decoding the new tokens alone drops
        # the space their first word-start piece stands for.
        prompt_text = self.tokenizer.decode(ids)
        return self.tokenizer.decode(ids + new)[len(prompt_text):]


MODELS = {lang: LanguageModels(lang) for lang in LANGS.values()}


# ---------------------------------------------------------------- free generation

def continue_text(language, prompt, max_new_tokens, temperature, top_p, repetition_penalty):
    lm = MODELS[LANGS[language]]
    if "lm" not in lm.models:
        return "(no pretrained checkpoint for this language)"
    sampling = {"temperature": temperature, "top_p": top_p, "repetition_penalty": repetition_penalty}
    return prompt.rstrip() + lm.generate("lm", prompt.rstrip(), int(max_new_tokens), **sampling)


# ---------------------------------------------------------------- reasoning

def new_question(language, hops):
    """A fresh test-style question: test-pool names, any paradigm, exactly `hops` hops."""
    lm = MODELS[LANGS[language]]
    gen = lm.gen
    seed = random.randrange(10**9)
    ex = gen.generate_examples(1, gen.TEST_ENTITY_POOL, gen.EVAL_PARADIGMS, "test", seed=seed,
                               n_entities=int(hops) + 1, exact_hops=int(hops))[0]
    state = {"text": ex.text, "query": list(ex.query), "answer_text": ex.answer_text,
             "relation": ex.answer_rel.split()[1], "paradigm": ex.paradigm}
    return ex.text, "", "", state


def asserted_relation(gen, answer: str, query) -> str | None:
    """Relation the answer sentence states between the queried pair, in query order."""
    flip = {">": "<", "<": ">", "=": "="}
    for a, rel, b in gen.parse_premises(answer):
        if (a, b) == tuple(query):
            return rel
        if (b, a) == tuple(query):
            return flip[rel]
    return None


def answer_question(language, question, state):
    lm = MODELS[LANGS[language]]
    if "reasoning" not in lm.models:
        return "(no reasoning checkpoint for this language)", ""
    answer = lm.generate("reasoning", question.strip(), 80, temperature=0.0).strip()
    if not state or state.get("text") != question:
        return answer, "Your own question: there is no solver answer to compare with."
    rel = asserted_relation(lm.gen, answer, state["query"])
    if rel is None:
        verdict = "✗ The answer does not state a relation between the two queried names."
    elif rel == state["relation"]:
        verdict = "✓ Correct."
    else:
        verdict = "✗ Wrong relation."
    return answer, f"{verdict}\nSolver: {state['answer_text']}  (paradigm: {state['paradigm']})"


# ---------------------------------------------------------------- UI

with gr.Blocks(title="Hindi & Assamese mini-LMs") as demo:
    gr.Markdown(
        "# Hindi & Assamese mini language models\n"
        "Two 25M-parameter Transformers per language, trained from scratch on ~500M tokens, "
        "running on CPU. **Generate** continues text with the pretrained Modern V2 model; "
        "**Reasoning** answers comparison questions with the Baseline V1 model fine-tuned on "
        "synthetic relational reasoning (trained on 1-3 hops; try longer chains)."
    )
    with gr.Tab("Generate"):
        with gr.Row():
            g_lang = gr.Radio(list(LANGS), value="Hindi", label="Language")
            g_tokens = gr.Slider(8, 128, value=48, step=8, label="New tokens")
        g_prompt = gr.Textbox(value=EXAMPLES["hindi"], lines=3, label="Prompt")
        with gr.Accordion("Sampling", open=False):
            g_temp = gr.Slider(0.0, 1.5, value=DEFAULT_SAMPLING["temperature"], step=0.05,
                               label="Temperature (0 = greedy)")
            g_top_p = gr.Slider(0.1, 1.0, value=DEFAULT_SAMPLING["top_p"], step=0.05, label="Top-p")
            g_rep = gr.Slider(1.0, 1.5, value=DEFAULT_SAMPLING["repetition_penalty"], step=0.05,
                              label="Repetition penalty")
        g_button = gr.Button("Generate", variant="primary")
        g_out = gr.Textbox(lines=6, label="Text")
        g_lang.change(lambda lang: EXAMPLES[LANGS[lang]], g_lang, g_prompt)
        g_button.click(continue_text, [g_lang, g_prompt, g_tokens, g_temp, g_top_p, g_rep], g_out)

    with gr.Tab("Reasoning"):
        r_state = gr.State({})
        with gr.Row():
            r_lang = gr.Radio(list(LANGS), value="Hindi", label="Language")
            r_hops = gr.Slider(2, 10, value=3, step=1, label="Hops between the queried people")
        r_new = gr.Button("New question")
        r_question = gr.Textbox(lines=5, label="Question (you can edit it)")
        r_button = gr.Button("Answer", variant="primary")
        r_answer = gr.Textbox(label="Model answer")
        r_check = gr.Textbox(lines=2, label="Check")
        r_new.click(new_question, [r_lang, r_hops], [r_question, r_answer, r_check, r_state])
        r_lang.change(new_question, [r_lang, r_hops], [r_question, r_answer, r_check, r_state])
        r_button.click(answer_question, [r_lang, r_question, r_state], [r_answer, r_check])
        demo.load(new_question, [r_lang, r_hops], [r_question, r_answer, r_check, r_state])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
