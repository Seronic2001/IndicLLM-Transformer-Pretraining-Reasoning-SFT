// Typing Hindi / Assamese in English letters: a romanised word ("bharat", "oxom") becomes its
// candidate spellings in the native script, ranked by the language's own tokenizer vocabulary
// (a common word is one vocabulary piece, a misspelling splits into several), so the ambiguous
// romanisation (a = अ/आ, t = त/ट, s = स/श/ष, ...) resolves to the spelling the corpus uses.
// Candidates are built in Devanagari; Assamese maps them to the Bengali-Assamese block.

const VIRAMA = "्";
const NASAL = new Set(["ं", "ँ"]);

// Latin -> consonant spellings, most likely first.
const HINDI_CONSONANTS = {
  chh: ["छ"], ksh: ["क्ष"], ck: ["क"], gy: ["ज्ञ", "ग्य"], kh: ["ख"], gh: ["घ"], ch: ["च", "छ"], jh: ["झ"],
  th: ["थ", "ठ"], dh: ["ध", "ढ"], ph: ["फ"], bh: ["भ"], sh: ["श", "ष"], rh: ["ढ़"],
  k: ["क"], g: ["ग"], c: ["च", "क"], j: ["ज"], z: ["ज़", "ज"], t: ["त", "ट"], d: ["द", "ड", "ड़"],
  n: ["न", "ण"], p: ["प"], f: ["फ़", "फ"], b: ["ब"], m: ["म"], y: ["य"], r: ["र"], l: ["ल"],
  v: ["व"], w: ["व"], s: ["स", "श", "ष"], h: ["ह"], q: ["क़", "क"], x: ["क्ष", "क्स"],
};
// Assamese: x and s are স/শ/ষ (the language pronounces them alike), j and z are জ/য,
// y between vowels is য়, v is ভ or ৱ.
const ASSAMESE_CONSONANTS = {
  ...HINDI_CONSONANTS,
  x: ["स", "श", "ष", "क्ष", "ख"], s: ["स", "श", "ष", "छ", "च"], ch: ["च", "छ"],
  z: ["ज", "य"], j: ["ज", "य"], y: ["य़", "य"],
  v: ["भ", "व"], f: ["फ"], q: ["क"], rh: ["ढ़", "ड़"],
};
// Latin -> [independent vowel, vowel sign after a consonant]
const HINDI_VOWELS = {
  aa: [["आ", "ा"]], ai: [["ऐ", "ै"], ["अई", "ई"]], au: [["औ", "ौ"], ["अउ", "उ"]],
  ee: [["ई", "ी"]], ii: [["ई", "ी"]], oo: [["ऊ", "ू"], ["ओ", "ो"]], uu: [["ऊ", "ू"]],
  ei: [["ए", "े"]], ou: [["औ", "ौ"], ["ओ", "ो"]],
  a: [["अ", ""], ["आ", "ा"]], i: [["इ", "ि"], ["ई", "ी"]], u: [["उ", "ु"], ["ऊ", "ू"]],
  e: [["ए", "े"]], o: [["ओ", "ो"]],
  // after a consonant "ri" is र + ि / ी (क्रि, मंत्री) or the vowel ऋ (कृष्ण); elsewhere रि (हरि)
  ri: [["रि", "्रि"], ["री", "्री"], ["ऋ", "ृ"]],
};
// Assamese writes its inherent vowel's sound as "o" (অসম = "oxom"), and ঐ as "oi".
const ASSAMESE_VOWELS = {
  ...HINDI_VOWELS, o: [["ओ", "ो"], ["अ", ""]], oi: [["अइ", "इ"], ["ऐ", "ै"], ["ओई", "ोई"]],
};
const TABLES = {
  hindi: { cons: HINDI_CONSONANTS, vowels: HINDI_VOWELS },
  assamese: { cons: ASSAMESE_CONSONANTS, vowels: ASSAMESE_VOWELS },
};
const KEYS = Object.fromEntries(Object.entries(TABLES).map(([lang, t]) =>
  [lang, Object.keys({ ...t.cons, ...t.vowels }).sort((a, b) => b.length - a.length)]));

// Devanagari -> Assamese: the Bengali block mirrors Devanagari at +0x80, except for ra/va
// (Assamese ৰ, ৱ) and letters the script lacks.
const TO_ASSAMESE = { "र": "ৰ", "व": "ৱ", "क़": "ক", "ज़": "জ", "फ़": "ফ" };
function toAssamese(s) {
  let out = "";
  for (const ch of s) {
    const cp = ch.codePointAt(0);
    out += TO_ASSAMESE[ch] ?? (cp >= 0x0900 && cp <= 0x097f ? String.fromCodePoint(cp + 0x80) : ch);
  }
  return out;
}

// A Latin word as units {kind, alts, prior}: "C" consonant, "V" vowel, and "J" between two
// consonants, joining them (virama) or keeping the unwritten-in-Latin inherent vowel
// (सरकार is typed "sarkar", not "sarakar"). prior[i] is a penalty, in pieces, for alts[i].
function parse(lang, word) {
  const { cons, vowels } = TABLES[lang], units = [];
  for (let i = 0; i < word.length;) {
    const key = KEYS[lang].find((k) => word.startsWith(k, i));
    if (!key) { i++; continue; } // a letter with no reading
    if (!vowels[key] && units.length && units[units.length - 1].kind === "C") units.push({ kind: "J", alts: [VIRAMA, ""] });
    units.push(vowels[key] ? { kind: "V", key, alts: vowels[key] } : { kind: "C", key, alts: [...cons[key]] });
    i += key.length;
  }
  // a typed word-final "a" is almost always ा (अच्छा, কথা); अ (धर्म) has to win on the vocabulary
  const last = units[units.length - 1];
  if (last && last.kind === "V" && last.key === "a" && units.length > 1) {
    units[units.length - 1] = { ...last, alts: [["आ", "ा"], ["अ", ""]], prior: [0, 0.5] };
  }
  // n / m closing a syllable (before a consonant or at the end) can be a nasal sign: हिंदी, मैं, माँ
  units.forEach((u, i) => {
    if (u.kind === "C" && (u.key === "n" || u.key === "m") && i > 0 && (i === units.length - 1 || units[i + 1].kind === "J")) {
      u.alts.push("ं", ...(u.key === "n" ? ["ँ"] : []));
    }
  });
  return units;
}

function render(units, choice) {
  let out = "", open = false, join = VIRAMA; // open: the last letter is a consonant still without a vowel
  units.forEach((u, i) => {
    const alt = u.alts[choice[i]];
    if (u.kind === "J") { join = alt; return; }
    if (u.kind === "V") { out += open ? alt[1] : alt[0]; open = false; return; }
    if (NASAL.has(alt)) { out += alt; open = false; return; }
    out += (open ? join : "") + alt;
    open = true; join = VIRAMA;
  });
  return out;
}

const LIMIT = 12000;
function choices(units) {
  const sizes = units.map((u) => u.alts.length);
  // too many combinations: keep only the first spellings of the units with the most, from the end
  while (sizes.reduce((a, b) => a * b, 1) > LIMIT) {
    const i = sizes.lastIndexOf(Math.max(...sizes));
    sizes[i]--;
  }
  let out = [[]];
  for (const n of sizes) out = out.flatMap((c) => Array.from({ length: n }, (_, k) => [...c, k]));
  return out;
}

// Up to `k` native spellings of one Latin word, best first.
export function suggest(lang, tok, word, k = 5) {
  const units = parse(lang, word.toLowerCase());
  if (!units.length) return [];
  const scored = new Map();
  for (const choice of choices(units)) {
    let text = render(units, choice);
    if (lang === "assamese") text = toAssamese(text);
    const ids = tok.encode(text);
    let bytes = 0, score = 0;
    for (const id of ids) { const [, s, type] = tok.pieces[id]; if (type === 6) bytes++; else score += s; }
    // fewest pieces (byte fallback counts triple, plus the priors), then the most frequent
    // pieces, then the most likely spelling of each letter
    const prior = units.reduce((p, u, i) => p + (u.prior ? u.prior[choice[i]] : 0), 0);
    if (!scored.has(text)) scored.set(text, [ids.length + 2 * bytes + prior, -score, choice.reduce((a, b) => a + b, 0)]);
  }
  return [...scored].sort(([, a], [, b]) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]).slice(0, k).map(([t]) => t);
}

// Every Latin word of `text` replaced by its best spelling.
export function transliterateAll(lang, tok, text) {
  return text.replace(/[A-Za-z]+/g, (w) => suggest(lang, tok, w, 1)[0] ?? w);
}

const LATIN_BEFORE_CARET = /[A-Za-z]+$/;
const COMMIT_KEYS = new Set([" ", ",", ".", "?", "!", ";", ":", "|"]);

// Live transliteration in a textarea: the Latin word before the caret is offered as native
// spellings in `bar`; Space or punctuation takes the first and is typed after it, Enter or
// Tab only takes it, a click or 1-5 picks another, Escape keeps the Latin. "." and "|" after
// a word become the danda "।".
export function attach(textarea, bar, { enabled, lang, tokenizer }) {
  let current = null; // {start, end, word, options}
  let keepLatin = null;
  const hint = () => {
    const ex = lang() === "assamese" ? "oxom → অসম, guwahati → গুৱাহাটী" : "bharat → भारत, rajdhani → राजधानी";
    bar.replaceChildren(Object.assign(document.createElement("span"), { className: "tl-hint",
      textContent: `Type in English letters: ${ex}. Space or Enter takes the first suggestion.` }));
    if (/[A-Za-z]/.test(textarea.value)) { // pasted or kept Latin text
      const all = Object.assign(document.createElement("button"), { type: "button", className: "tl-convert", textContent: "Convert English words" });
      all.addEventListener("click", () => convertAll());
      bar.append(all);
    }
  };
  const show = () => {
    bar.hidden = !enabled() || textarea.hidden;
    if (bar.hidden) return;
    if (!current || !current.options.length) { hint(); return; }
    const items = current.options.map((t, i) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = `tl-opt${i === 0 ? " first" : ""}`;
      b.append(Object.assign(document.createElement("kbd"), { textContent: String(i + 1) }), t);
      b.addEventListener("mousedown", (e) => { e.preventDefault(); commit(t); });
      return b;
    });
    const keep = document.createElement("button");
    keep.type = "button"; keep.className = "tl-opt latin"; keep.textContent = current.word;
    keep.title = "Keep the English letters (Esc)";
    keep.addEventListener("mousedown", (e) => { e.preventDefault(); keepLatin = current.word; current = null; show(); });
    bar.replaceChildren(...items, keep);
  };
  const update = async () => {
    if (!enabled()) { current = null; show(); return; }
    const caret = textarea.selectionStart;
    const m = textarea.selectionStart === textarea.selectionEnd && LATIN_BEFORE_CARET.exec(textarea.value.slice(0, caret));
    if (!m || m[0] === keepLatin) { current = null; show(); return; }
    const tok = await tokenizer(lang());
    current = { start: caret - m[0].length, end: caret, word: m[0], options: suggest(lang(), tok, m[0]) };
    show();
  };
  const commit = (text) => {
    if (!current) return;
    textarea.setRangeText(text, current.start, current.end, "end");
    current = null; keepLatin = null;
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    textarea.focus();
  };
  textarea.addEventListener("input", update);
  textarea.addEventListener("click", update);
  textarea.addEventListener("keydown", (e) => {
    if (!enabled() || !current || e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
    if (e.key === "Escape") { keepLatin = current.word; current = null; show(); e.preventDefault(); return; }
    if (/^[1-5]$/.test(e.key) && current.options[Number(e.key) - 1]) {
      e.preventDefault(); commit(current.options[Number(e.key) - 1]); return;
    }
    // Enter and Tab only accept the suggestion (a second Enter starts a new line)
    if ((e.key === "Enter" || e.key === "Tab") && current.options.length) {
      e.preventDefault(); commit(current.options[0]); return;
    }
    if (COMMIT_KEYS.has(e.key) && current.options.length) {
      commit(current.options[0]);
      if (e.key === "." || e.key === "|") { e.preventDefault(); textarea.setRangeText("।", textarea.selectionStart, textarea.selectionEnd, "end"); textarea.dispatchEvent(new Event("input", { bubbles: true })); }
    }
  });
  async function convertAll() {
    const tok = await tokenizer(lang());
    textarea.value = transliterateAll(lang(), tok, textarea.value);
    current = null; keepLatin = null;
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
  }
  return { refresh: update, convertAll };
}
