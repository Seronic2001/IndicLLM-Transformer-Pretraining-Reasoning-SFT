// UI of the static demo: loads models lazily and runs them with lm.js.
import { Tokenizer, Model, assertedRelation, splitRationale } from "./lm.js";
import { attach } from "./translit.js";

const ort = window.ort;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const LANGS = { hindi: { code: "hi", other: "assamese", abc: "अबक" }, assamese: { code: "as", other: "hindi", abc: "অবক" } };
const PROMPTS = {
  hindi: ["भारत की राजधानी नई दिल्ली है और", "आज सुबह से ही तेज़ बारिश", "क्रिकेट विश्व कप के फाइनल में", "प्रधानमंत्री ने कहा कि"],
  assamese: ["অসমৰ ৰাজধানী গুৱাহাটী আৰু", "আজি পুৱাৰ পৰাই প্ৰবল বৰষুণ", "বিহু উৎসৱৰ সময়ত", "মুখ্যমন্ত্ৰীয়ে কয় যে"],
};
const TOK_TEXTS = {
  hindi: ["भारत की राजधानी नई दिल्ली है।", "मैंने आज Python में कोड लिखा।", "२०२६ में 1,234 लोग आए।", "অসমীয়া লিপি"],
  assamese: ["অসমৰ ৰাজধানী গুৱাহাটী।", "মই আজি Python ত code লিখিলোঁ।", "২০২৬ চনত 1,234 জন মানুহ আহিছিল।", "हिन्दी लिपि"],
};
const PRESETS = {
  greedy: { temp: 0, topp: 1, rep: 1.15 },
  balanced: { temp: 0.8, topp: 0.9, rep: 1.15 }, // the Phase 2 evaluation setting
  creative: { temp: 1.1, topp: 0.95, rep: 1.2 },
};
const HOPS = [2, 3, 4, 5, 6, 8, 10];
const TRAINED_HOPS = 3;

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const typers = []; // transliteration helpers of the text boxes, set up at the end
const state = { lang: "hindi", hops: 3, variant: "v1_direct", question: null, score: {}, stop: false, drafts: {} };

// ---- lazy loading
const cache = new Map(); // "tokenizer:<lang>", "questions:<lang>", "model:<lang>_<role>"
function once(key, load) {
  if (!cache.has(key)) cache.set(key, load().catch((e) => { cache.delete(key); throw e; }));
  return cache.get(key);
}
const getJSON = (url) => fetch(url).then((r) => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); });
const tokenizer = (lang) => once(`tokenizer:${lang}`, async () => new Tokenizer(await getJSON(`data/${lang}_tokenizer.json`)));
const questions = (lang) => once(`questions:${lang}`, () => getJSON(`data/${lang}_questions.json`));

function model(name, progress) {
  return once(`model:${name}`, async () => {
    const meta = await getJSON(`models/${name}.json`);
    const res = await fetch(`models/${name}.onnx`);
    if (!res.ok) throw new Error(`models/${name}.onnx: ${res.status}`);
    const total = Number(res.headers.get("content-length")) || 32e6;
    const reader = res.body.getReader(), chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.length;
      progress(`Downloading model · ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`, Math.min(1, got / total));
    }
    const bytes = new Uint8Array(got);
    let at = 0;
    for (const c of chunks) { bytes.set(c, at); at += c.length; }
    progress("Starting model…", 1);
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    return new Model(ort, session, meta);
  });
}

// Status line with an optional progress bar; frac null hides the bar.
function statusOf(prefix) {
  const bar = $(`${prefix}-bar`), text = $(`${prefix}-status`);
  return (msg, frac = null) => {
    text.textContent = msg;
    bar.hidden = frac === null;
    if (frac !== null) bar.firstElementChild.style.width = `${(frac * 100).toFixed(1)}%`;
  };
}

async function busy(buttons, status, work) {
  buttons.forEach((b) => { b.disabled = true; });
  try { await work(status); } catch (e) { status(`Error: ${e.message}`); console.error(e); }
  finally { buttons.forEach((b) => { b.disabled = false; }); }
}

function radios(group, onPick) {
  group.addEventListener("click", (e) => {
    const b = e.target.closest("button[role=radio]");
    if (!b) return;
    for (const x of group.querySelectorAll("button[role=radio]")) x.setAttribute("aria-checked", String(x === b));
    onPick(b);
  });
}
const pickRadio = (group, attr, value) => {
  for (const x of group.querySelectorAll("button[role=radio]")) x.setAttribute("aria-checked", String(x.dataset[attr] === String(value)));
};

function chips(container, texts, onPick) {
  container.replaceChildren(...texts.map((t) => {
    const c = el("button", "chip", t); c.type = "button"; c.title = t;
    c.addEventListener("click", () => onPick(t));
    return c;
  }));
}
const stat = (t) => el("span", "stat", t);
const channel = new MessageChannel();
const nextTask = () => new Promise((r) => { channel.port1.onmessage = () => r(); channel.port2.postMessage(0); });

// ---- tabs
const TABS = [["tab-gen", "gen"], ["tab-rsn", "rsn"], ["tab-tok", "tok"]];
for (const [tab] of TABS) {
  $(tab).addEventListener("click", () => {
    for (const [t, p] of TABS) { $(t).setAttribute("aria-selected", String(t === tab)); $(p).hidden = t !== tab; }
    if (tab === "tab-tok") renderTokens();
  });
}

// ---- language
radios($("lang"), (b) => setLang(b.dataset.lang));
function setLang(lang) {
  const prev = state.lang;
  state.lang = lang;
  for (const id of ["prompt", "question", "tok-in", "gen-out", "rsn-results", "premises", "ask", "tok-out"]) $(id).lang = LANGS[lang].code;
  $("translit-example").textContent = `(abc → ${LANGS[lang].abc})`;
  // each language keeps its own prompt and tokenizer text; switching back restores them
  if (prev !== lang) state.drafts[prev] = { prompt: $("prompt").value, tok: $("tok-in").value };
  const draft = state.drafts[lang] || { prompt: PROMPTS[lang][0], tok: TOK_TEXTS[lang][0] };
  $("prompt").value = draft.prompt; $("tok-in").value = draft.tok;
  chips($("gen-examples"), PROMPTS[lang], (t) => { $("prompt").value = t; $("prompt").focus(); });
  chips($("tok-examples"), TOK_TEXTS[lang], (t) => { $("tok-in").value = t; renderTokens(); });
  if (prev !== lang) { clearOutput($("gen-out")); $("gen-stats").replaceChildren(); $("gen-copy").hidden = true; }
  renderModels();
  newQuestion();
  typers.forEach((t) => t.refresh());
  if (!$("tok").hidden) renderTokens();
}
function clearOutput(out) { out.replaceChildren(); out.classList.add("empty"); }

// ---- generate
for (const id of ["tokens", "temp", "topp", "rep"]) {
  const show = () => { $(`${id}-v`).textContent = $(id).value; };
  $(id).addEventListener("input", () => { show(); if (id !== "tokens") pickRadio($("preset"), "preset", ""); });
  show();
}
radios($("preset"), (b) => {
  for (const [id, v] of Object.entries(PRESETS[b.dataset.preset])) { $(id).value = v; $(`${id}-v`).textContent = $(id).value; }
});

$("gen-stop").addEventListener("click", () => { state.stop = true; });
$("gen-copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("gen-out").textContent); $("gen-copy").textContent = "Copied"; }
  catch { $("gen-copy").textContent = "Copy failed"; }
  setTimeout(() => { $("gen-copy").textContent = "Copy"; }, 1500);
});

$("gen-go").addEventListener("click", () => busy([$("gen-go")], statusOf("gen"), async (status) => {
  const lang = state.lang, prompt = $("prompt").value.trim();
  if (!prompt) { status("Type a prompt first."); return; }
  const tok = await tokenizer(lang);
  const lm = await model(`${lang}_lm`, status);
  const ids = tok.encode(prompt), promptText = tok.decode(ids);
  const out = $("gen-out"), p = el("span", "p", promptText), g = el("span", "g"), caret = el("span", "caret");
  out.classList.remove("empty"); out.replaceChildren(p, g, caret);
  $("gen-stats").replaceChildren(); $("gen-copy").hidden = true;
  state.stop = false; $("gen-stop").hidden = false;
  status("Generating…");
  const t0 = performance.now();
  let added;
  try {
    added = await lm.generate(ids, Number($("tokens").value), tok.eosId, {
      temperature: Number($("temp").value), topP: Number($("topp").value), repetitionPenalty: Number($("rep").value),
    }, async (sofar) => {
      // decode prompt and continuation together, so the first word's leading space survives
      g.textContent = tok.decode([...ids, ...sofar]).slice(promptText.length);
      await nextTask(); // the WASM steps chain as microtasks: let the page paint and see Stop
      return state.stop;
    });
  } finally { caret.remove(); $("gen-stop").hidden = true; }
  const secs = (performance.now() - t0) / 1000;
  const max = Number($("tokens").value);
  $("gen-stats").replaceChildren(stat(`${ids.length} prompt tok`), stat(`${added.length} new tok`), stat(`${secs.toFixed(1)} s`),
    stat(`${(added.length / secs).toFixed(0)} tok/s`));
  $("gen-copy").hidden = false;
  status(state.stop ? "Stopped." : added.length < max ? "The model ended the text (end-of-sequence token)." : "");
}));

// ---- reasoning
const ARCH = { v1: "Baseline V1 (learned positions, GELU, LayerNorm)", v2: "Modern V2 (RoPE, SwiGLU, RMSNorm)" };
const MODE = { direct: "answers in one sentence", cot: "first writes the chain of comparisons (chain of thought), then answers" };
const CHANCE = 0.481; // always the most common relation, on the test split
const MAX_ANSWER_TOKENS = 80; // the evaluation's limit
const variantName = (v) => `${v.arch.toUpperCase()} ${v.mode === "cot" ? "CoT" : "Direct"}`;
const pct = (x) => `${(x * 100).toFixed(1)}%`;
// data/models.json lists each language's fine-tuned variants; older builds have only V1 Direct
const catalog = (lang) => once(`catalog:${lang}`, () => getJSON("data/models.json")
  .then((c) => c[lang]).catch(() => [{ id: "v1_direct", arch: "v1", mode: "direct" }]));
const variantModel = (lang, id, progress) => model(`${lang}_${id}`, progress).catch((e) =>
  id === "v1_direct" ? model(`${lang}_reasoning`, progress) : Promise.reject(e));

const hopGroup = $("hops");
hopGroup.replaceChildren(...HOPS.map((h) => {
  const b = el("button", "", String(h)); b.setAttribute("role", "radio"); b.dataset.hops = h;
  b.setAttribute("aria-checked", String(h === state.hops));
  return b;
}));
radios(hopGroup, (b) => { state.hops = Number(b.dataset.hops); renderModels(); newQuestion(); });
radios($("variant"), (b) => { state.variant = b.dataset.variant; renderModels(); resetAnswer(); });

function hopNote() {
  const note = $("hop-note"), beyond = state.hops > TRAINED_HOPS;
  note.textContent = beyond ? `Beyond the 1–${TRAINED_HOPS} hops seen in training` : "Within the training range";
  note.className = `hop-note ${beyond ? "beyond" : "muted"}`;
}

// Model picker, its description and the held-out accuracy of every variant at the chosen depth.
async function renderModels() {
  hopNote();
  const variants = await catalog(state.lang);
  if (!variants.some((v) => v.id === state.variant) && state.variant !== "all") state.variant = variants[0].id;
  const group = $("variant");
  const options = [...variants.map((v) => [v.id, variantName(v)]), ...(variants.length > 1 ? [["all", "Compare all"]] : [])];
  group.replaceChildren(...options.map(([id, label]) => {
    const b = el("button", "", label); b.setAttribute("role", "radio"); b.dataset.variant = id;
    b.setAttribute("aria-checked", String(id === state.variant));
    return b;
  }));
  const note = $("variant-note");
  const v = variants.find((x) => x.id === state.variant);
  if (v) {
    note.replaceChildren(el("b", "", `${ARCH[v.arch]}`), `, fine-tuned to ${v.mode === "cot" ? "reason step by step" : "answer directly"}: it ${MODE[v.mode]}.`);
  } else {
    const mb = variants.reduce((n, x) => n + (x.mb || 32), 0);
    note.replaceChildren(`Runs the question through all ${variants.length} fine-tuned models, one after another. `,
      el("b", "", `The first run downloads about ${Math.round(mb)} MB.`));
  }
  const scored = variants.filter((x) => x.accuracy && x.accuracy[state.hops] != null);
  $("acc").hidden = !scored.length;
  if (!scored.length) return;
  $("acc-title").textContent = `Measured accuracy at ${state.hops} hops`;
  const split = state.hops === 2 ? "the main test split (2,000 questions, mostly 2–3 hops)" : `500 held-out ${state.hops}-hop questions`;
  $("acc-foot").textContent = `On ${split}. Dashed line: chance (${pct(CHANCE)}).`;
  $("acc-bars").replaceChildren(...scored.map((x) => {
    const a = x.accuracy[state.hops];
    const row = el("div", `acc-row${x.id === state.variant || state.variant === "all" ? " sel" : ""}`);
    row.setAttribute("role", "listitem");
    row.title = `${variantName(x)}: ${pct(a)} correct at ${state.hops} hops${x.job ? ` (job ${x.job})` : ""}`;
    const track = el("div", "acc-track"), bar = el("i"), chance = el("span", "chance");
    bar.style.width = `${a * 100}%`; chance.style.left = `${CHANCE * 100}%`;
    track.append(bar, chance);
    row.append(el("span", "n", variantName(x)), track, el("span", "v", pct(a)));
    return row;
  }));
}

// The question as one list item per sentence, the two queried names marked.
function markNames(s, names) {
  const re = names.length ? new RegExp(`(${names.map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "g") : null;
  const span = el("span");
  for (const part of re ? s.split(re) : [s]) {
    if (!part) continue;
    const i = names.indexOf(part);
    span.append(i < 0 ? document.createTextNode(part) : el("mark", i === 0 ? "a" : "b", part));
  }
  return span;
}
function renderPremises(text, query) {
  const names = (query || []).filter(Boolean);
  const sentences = text.split(/(?<=[।?])\s*/).map((s) => s.trim()).filter(Boolean);
  $("premises").replaceChildren(...sentences.filter((s) => !s.endsWith("?")).map((s) => {
    const li = el("li"); li.append(markNames(s, names)); return li;
  }));
  $("ask").replaceChildren(...sentences.filter((s) => s.endsWith("?")).map((s) => markNames(s, names)));
}

function resetAnswer() {
  clearOutput($("rsn-results")); $("rsn-solver").hidden = true; $("rsn-stats").replaceChildren(); statusOf("rsn")("");
}

let questionToken = 0;
async function newQuestion() {
  const lang = state.lang, hops = state.hops, mine = ++questionToken;
  let pool;
  try { pool = (await questions(lang)).filter((q) => q.hops === hops); }
  catch (e) { statusOf("rsn")(`Error: ${e.message}`); return; }
  if (mine !== questionToken) return; // a newer request won
  state.question = pool[Math.floor(Math.random() * pool.length)];
  $("question").value = state.question.text;
  renderPremises(state.question.text, state.question.query);
  setEditing(false);
  resetAnswer();
}
$("rsn-new").addEventListener("click", newQuestion);

function setEditing(on) {
  $("question").hidden = !on; $("qview").hidden = on;
  $("rsn-edit").textContent = on ? "Done editing" : "Edit question";
  typers.forEach((t) => t.refresh());
  if (on) $("question").focus();
  else {
    const text = $("question").value.trim(), q = state.question;
    renderPremises(text, q && q.text === text ? q.query : null);
  }
}
$("rsn-edit").addEventListener("click", () => setEditing($("question").hidden));
$("question").addEventListener("input", resetAnswer);

// One result card: model name, verdict, the chain of thought (CoT models) and the answer.
function resultCard(v) {
  const card = el("div", "res none"), head = el("div", "res-head"), body = el("div");
  const acc = v.accuracy && v.accuracy[state.hops];
  head.append(el("span", "who", variantName(v)), ...(acc != null ? [el("span", "tag", `${pct(acc)} at ${state.hops} hops`)] : []),
    el("span", "spacer"));
  body.append(el("span", "res-pending", "waiting…"));
  card.append(head, body);
  return {
    card,
    pending: (t) => { body.replaceChildren(el("span", "res-pending", t)); },
    show: (output, verdict, secs, nTokens) => {
      const r = splitRationale(output);
      const parts = [];
      if (v.mode === "cot" || r.steps.length) {
        const chain = el("div", "chain");
        chain.append(el("span", "chain-label", "Reasoning"));
        r.steps.forEach((s, i) => { if (i) chain.append(el("span", "so", "and")); chain.append(el("span", "step", s)); });
        if (r.conclusion) chain.append(el("span", "so", "so"), el("span", "step concl", r.conclusion));
        if (!r.steps.length) chain.append(el("span", "so", "(no reasoning written)"));
        parts.push(chain);
      }
      const truncated = !r.closed || (nTokens >= MAX_ANSWER_TOKENS && !r.answer);
      parts.push(el("div", "res-text", r.answer || (truncated ? `(stopped at the ${MAX_ANSWER_TOKENS}-token limit before answering)` : "(empty answer)")));
      body.replaceChildren(...parts);
      if (verdict) {
        card.className = `res ${verdict.ok ? "ok" : "bad"}`;
        head.append(el("span", verdict.ok ? "ok" : "bad", verdict.text));
      }
      head.append(el("span", "stat", `${nTokens} tok · ${secs.toFixed(2)} s`));
    },
  };
}

function scoreLine() {
  const parts = Object.entries(state.score).filter(([, s]) => s.n).map(([name, s]) => `${name} ${s.ok}/${s.n}`);
  $("rsn-score").textContent = parts.length ? `This session: ${parts.join(" · ")}` : "";
}

$("rsn-go").addEventListener("click", () => busy([$("rsn-go"), $("rsn-new")], statusOf("rsn"), async (status) => {
  if (!$("question").hidden) setEditing(false);
  const lang = state.lang, text = $("question").value.trim();
  if (!text) { status("Write a question first."); return; }
  const variants = await catalog(lang);
  const chosen = state.variant === "all" ? variants : variants.filter((v) => v.id === state.variant);
  const tok = await tokenizer(lang), q = state.question, ids = tok.encode(text);
  const own = !q || q.text !== text;
  const solver = $("rsn-solver");
  solver.hidden = false;
  solver.replaceChildren(...(own ? [el("span", "", "Your own question: there is no solver answer to compare with.")]
    : [el("span", "", "Solver:"), el("span", "solver", q.answer), el("span", "tag", q.paradigm.replaceAll("_", " "))]));
  const cards = chosen.map((v) => resultCard(v));
  const results = $("rsn-results");
  results.classList.remove("empty"); results.replaceChildren(...cards.map((c) => c.card));
  $("rsn-stats").replaceChildren(stat(`${ids.length} question tok`));
  for (const [i, v] of chosen.entries()) {
    const progress = (t, f) => { status(`${variantName(v)}: ${t}`, f); cards[i].pending(t); };
    const lm = await variantModel(lang, v.id, progress);
    progress("answering…", null);
    await nextTask(); // paint the status before the WASM steps run
    const t0 = performance.now();
    const added = await lm.generate(ids, MAX_ANSWER_TOKENS, tok.eosId, { temperature: 0 });
    const secs = (performance.now() - t0) / 1000, output = tok.decode(added).trim();
    let verdict = null;
    if (!own) {
      const rel = assertedRelation(lang, splitRationale(output).answer, q.query), ok = rel === q.relation;
      verdict = { ok, text: ok ? "✓ Correct" : rel === null ? "✗ No comparison of the two names" : "✗ Wrong relation" };
      const s = (state.score[variantName(v)] ||= { ok: 0, n: 0 });
      s.ok += ok; s.n++;
    }
    cards[i].show(output, verdict, secs, added.length);
  }
  scoreLine();
  status("");
}));

// ---- tokenizer
let tokToken = 0;
async function renderTokens() {
  const lang = state.lang, text = $("tok-in").value.slice(0, 4000), mine = ++tokToken;
  const tok = await tokenizer(lang);
  if (mine !== tokToken) return;
  const ids = tok.encode(text), segs = tok.segments(ids);
  let colour = 0;
  $("tok-out").replaceChildren(...segs.map((s) => {
    const nl = s.kind === "bytes" && /^\s+$/.test(s.text);
    const cls = s.kind === "bytes" ? (nl ? "nl" : "bytes") : `c${colour++ % 5}`;
    const t = el("span", `tk ${cls}${s.wordStart ? " w" : ""}`);
    t.title = `${s.kind === "bytes" ? `${s.ids.length} byte${s.ids.length > 1 ? "s" : ""}` : `piece ${JSON.stringify(tok.pieces[s.ids[0]][0])}`} · id ${s.ids.join(" ")}`;
    t.append(el("span", "s", nl ? s.text.replaceAll("\n", "↵").replaceAll("\t", "⇥") : s.text), el("span", "i", s.ids.join(" ")));
    return t;
  }));
  const chars = Array.from(text.trim()).length, words = text.trim() ? text.trim().split(/\s+/).length : 0;
  const bytes = segs.filter((s) => s.kind === "bytes").reduce((n, s) => n + s.ids.length, 0);
  const metric = (v, k) => { const m = el("div", "metric"); m.append(el("div", "v", v), el("div", "k", k)); return m; };
  const cells = [metric(ids.length, "tokens"), metric(chars, "characters"), metric(words, "words"),
    metric(words ? (ids.length / words).toFixed(2) : "–", "tokens per word"), metric(bytes, "byte-fallback tokens")];
  $("tok-stats").replaceChildren(...cells);
  const other = LANGS[lang].other;
  try {
    const n = (await tokenizer(other)).encode(text).length;
    if (mine === tokToken) $("tok-stats").append(metric(n, `with the ${other[0].toUpperCase() + other.slice(1)} tokenizer`));
  } catch { /* the comparison is optional */ }
}
let tokTimer = 0;
$("tok-in").addEventListener("input", () => { clearTimeout(tokTimer); tokTimer = setTimeout(renderTokens, 120); });
$("tok-ids").addEventListener("change", () => $("tok-out").classList.toggle("ids", $("tok-ids").checked));

// ---- typing in English letters (translit.js)
const translitOn = () => $("translit").checked;
try { $("translit").checked = localStorage.getItem("translit") === "1"; } catch { /* per-viewer convenience only */ }
typers.push(...["prompt", "question", "tok-in"].map((id) =>
  attach($(id), $(`${id}-tl`), { enabled: translitOn, lang: () => state.lang, tokenizer })));
$("translit").addEventListener("change", () => {
  try { localStorage.setItem("translit", translitOn() ? "1" : "0"); } catch { /* ignore */ }
  typers.forEach((t) => t.refresh());
});

setLang("hindi");
