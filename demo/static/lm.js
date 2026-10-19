// In-browser inference for the Hindi / Assamese models: SentencePiece BPE tokenizer,
// cached generation with ONNX Runtime Web, and the reasoning-answer check.
// Runs unchanged in the browser and in Node (test_static.mjs); `ort` is passed in.

const WORD = "▁"; // SentencePiece word-start marker
const NORMAL = 1, UNKNOWN = 2, CONTROL = 3, BYTE = 6;

export class Tokenizer {
  // json: {pieces: [[piece, score, type], ...], unk_id, eos_id} (build_static.tokenizer_json)
  constructor(json) {
    this.pieces = json.pieces;
    this.unkId = json.unk_id;
    this.eosId = json.eos_id;
    this.ids = new Map();
    this.byteIds = new Array(256);
    json.pieces.forEach(([piece, , type], id) => {
      if (type === BYTE) this.byteIds[parseInt(piece.slice(3, 5), 16)] = id;
      else if (type === NORMAL) this.ids.set(piece, id);
    });
  }

  // nfkc normalizer with remove_extra_whitespaces (runs of spaces only: tabs and newlines
  // are kept and become byte pieces, as in sentencepiece), escape_whitespaces, add_dummy_prefix
  normalize(text) {
    const s = text.normalize("NFKC").replace(/ +/g, " ").replace(/^ | $/g, "");
    return s ? WORD + s.replaceAll(" ", WORD) : "";
  }

  // BPE: repeatedly merge the adjacent pair whose concatenation is a vocabulary piece with
  // the highest score (leftmost on ties); symbols left outside the vocabulary fall back to bytes.
  encode(text) {
    const symbols = Array.from(this.normalize(text));
    for (;;) {
      let best = -Infinity, at = -1;
      for (let i = 0; i + 1 < symbols.length; i++) {
        const id = this.ids.get(symbols[i] + symbols[i + 1]);
        if (id !== undefined && this.pieces[id][1] > best) { best = this.pieces[id][1]; at = i; }
      }
      if (at < 0) break;
      symbols.splice(at, 2, symbols[at] + symbols[at + 1]);
    }
    const out = [];
    for (const s of symbols) {
      const id = this.ids.get(s);
      if (id !== undefined) out.push(id);
      else for (const b of new TextEncoder().encode(s)) out.push(this.byteIds[b] ?? this.unkId);
    }
    return out;
  }

  decode(ids) {
    let text = "", bytes = [];
    const flush = () => { if (bytes.length) { text += new TextDecoder().decode(new Uint8Array(bytes)); bytes = []; } };
    for (const id of ids) {
      const [piece, , type] = this.pieces[id];
      if (type === BYTE) { bytes.push(parseInt(piece.slice(3, 5), 16)); continue; }
      flush();
      if (type === CONTROL) continue;
      text += type === UNKNOWN ? " ⁇ " : piece.replaceAll(WORD, " ");
    }
    flush();
    return text.startsWith(" ") ? text.slice(1) : text;
  }

  // ids as display segments {text, ids, kind}: kind "piece" (wordStart when it begins with
  // the word marker, which is dropped from text), "bytes" (a run of byte-fallback pieces,
  // decoded together), "control" or "unknown".
  segments(ids) {
    const out = [];
    for (const id of ids) {
      const [piece, , type] = this.pieces[id];
      const last = out[out.length - 1];
      if (type === BYTE) {
        if (last && last.kind === "bytes" && !last.complete) last.ids.push(id);
        else out.push({ kind: "bytes", ids: [id] });
        const seg = out[out.length - 1];
        const bytes = seg.ids.map((b) => parseInt(this.pieces[b][0].slice(3, 5), 16));
        seg.text = new TextDecoder().decode(new Uint8Array(bytes));
        seg.complete = !seg.text.includes("�");
        continue;
      }
      const kind = type === CONTROL ? "control" : type === UNKNOWN ? "unknown" : "piece";
      out.push({ kind, ids: [id], wordStart: piece.startsWith(WORD), text: piece.replaceAll(WORD, " ").trimStart() || " " });
    }
    return out;
  }
}

// One loaded ONNX model; meta = {n_layer, n_head, d_k, block_size} (models/<name>.json).
export class Model {
  constructor(ort, session, meta) { this.ort = ort; this.session = session; this.meta = meta; }

  emptyCache() {
    const { n_layer, n_head, d_k } = this.meta;
    return new this.ort.Tensor("float32", new Float32Array(0), [n_layer, 1, n_head, 0, d_k]);
  }

  // Extend ids by up to maxNew tokens; stops at eos. sampling: {temperature, topP,
  // repetitionPenalty}, temperature 0 = greedy. onToken(newIds) is called after each token;
  // returning (or resolving to) true stops generation.
  async generate(ids, maxNew, eosId, sampling = {}, onToken = null) {
    const { temperature = 0, topP = 1, repetitionPenalty = 1 } = sampling;
    const room = this.meta.block_size - maxNew;
    let prompt = ids.slice(-Math.max(1, room));
    let pastK = this.emptyCache(), pastV = this.emptyCache();
    const seq = [...prompt], added = [];
    let step = prompt;
    for (let n = 0; n < maxNew; n++) {
      const feeds = {
        input_ids: new this.ort.Tensor("int64", BigInt64Array.from(step.map(BigInt)), [1, step.length]),
        past_k: pastK, past_v: pastV,
      };
      const out = await this.session.run(feeds);
      pastK = out.present_k; pastV = out.present_v;
      const next = pick(Float32Array.from(out.logits.data), seq, temperature, topP, repetitionPenalty);
      if (next === eosId) break;
      seq.push(next); added.push(next); step = [next];
      if (onToken && await onToken(added)) break;
    }
    return added;
  }
}

// Same rules as model.generate in the Python code: repetition penalty on the last 64
// tokens, then greedy (temperature 0) or temperature + nucleus (top-p) sampling.
function pick(logits, seq, temperature, topP, repetitionPenalty) {
  if (repetitionPenalty > 1) {
    for (const t of new Set(seq.slice(-64))) {
      logits[t] = logits[t] > 0 ? logits[t] / repetitionPenalty : logits[t] * repetitionPenalty;
    }
  }
  if (!(temperature > 1e-5)) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) { logits[i] /= temperature; if (logits[i] > max) max = logits[i]; }
  const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
  const probs = order.map((i) => Math.exp(logits[i] - max));
  const total = probs.reduce((a, b) => a + b, 0);
  let keep = 0, cum = 0;
  while (keep < order.length && (keep === 0 || cum <= topP * total)) { cum += probs[keep]; keep++; }
  let r = Math.random() * cum;
  for (let j = 0; j < keep; j++) { r -= probs[j]; if (r <= 0) return order[j]; }
  return order[keep - 1];
}

// The relation an answer sentence states between the queried pair (in query order), as in
// generate_reasoning.parse_premises; null when the answer does not compare the two names.
const ANSWER_PATTERNS = {
  hindi: [
    [/([^\s।:?]+) की (\S+) ([^\s।:?]+) की (\S+) से (अधिक|कम) (नहीं )?है/g, (m) => strict(m[5] === "अधिक", !!m[6])],
    [/([^\s।:?]+) की (\S+) ([^\s।:?]+) की (\S+) के बराबर (नहीं )?है/g, (m) => (m[5] ? null : "=")],
  ],
  assamese: [
    [/([^\s।?]+)ৰ (\S+) ([^\s।?]+)ৰ (\S+)তকৈ (বেছি|কম)( নহয়)?/g, (m) => strict(m[5] === "বেছি", !!m[6])],
    [/([^\s।?]+)ৰ (\S+) ([^\s।?]+)ৰ (\S+)ৰ সমান( নহয়)?/g, (m) => (m[5] ? null : "=")],
  ],
};
const strict = (more, negated) => ((more !== negated) ? ">" : "<");
const FLIP = { ">": "<", "<": ">", "=": "=" };

// A chain-of-thought output "[कारण: a > b और b > c इसलिए a > c] <answer>" (Assamese: কাৰণ, আৰু,
// সেয়ে) as {steps, conclusion, answer, closed}; the answer is what the evaluation scores (the
// text after the rationale, finetune.extract_answer_from_prediction). Direct output has no steps.
const RATIONALE = /\[(?:कारण|কাৰণ|কারণ):\s*([^\]]*)(\]?)/;
export function splitRationale(text) {
  const m = text.match(RATIONALE);
  if (!m) return { steps: [], conclusion: null, answer: text.trim(), closed: true };
  const [chain, conclusion = null] = m[1].trim().split(/\s+(?:इसलिए|সে(?:য়|য়)ে)\s+/);
  return {
    steps: chain ? chain.split(/\s+(?:और|আৰু)\s+/).filter(Boolean) : [],
    conclusion, answer: text.replace(m[0], "").trim(), closed: m[2] === "]",
  };
}

export function assertedRelation(lang, answer, query) {
  for (const [re, relation] of ANSWER_PATTERNS[lang]) {
    for (const m of answer.matchAll(re)) {
      const rel = relation(m);
      if (rel === null) continue;
      if (m[1] === query[0] && m[3] === query[1]) return rel;
      if (m[1] === query[1] && m[3] === query[0]) return FLIP[rel];
    }
  }
  return null;
}
