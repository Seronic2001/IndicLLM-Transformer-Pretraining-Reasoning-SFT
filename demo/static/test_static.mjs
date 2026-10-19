// Test the static demo's lm.js in Node with ONNX Runtime Web (the browser's WASM runtime).
// usage (from a directory where `npm install onnxruntime-web@1.20.1` was run):
//   node <repo>/demo/static/test_static.mjs <built static dir> <tokenizer fixture.json>
// The fixture maps lang -> [{text, ids, decoded}] from Python sentencepiece; every string
// must encode to the same ids and decode to the same text. Then every reasoning variant in
// data/models.json answers 2 bank questions per hop count, and the LM continues a prompt.
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { Tokenizer, Model, assertedRelation, splitRationale } from "./lm.js";

const [built, fixturePath] = process.argv.slice(2);
const ort = await import(pathToFileURL(createRequire(path.join(process.cwd(), "x.js")).resolve("onnxruntime-web")));
const read = (p) => JSON.parse(fs.readFileSync(path.join(built, p), "utf8"));
const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
let failed = false;

async function load(name) {
  const session = await ort.InferenceSession.create(fs.readFileSync(path.join(built, "models", `${name}.onnx`)));
  return new Model(ort, session, read(`models/${name}.json`));
}

for (const lang of ["hindi", "assamese"]) {
  const tok = new Tokenizer(read(`data/${lang}_tokenizer.json`));
  let idMismatch = 0, textMismatch = 0;
  for (const { text, ids, decoded } of fixture[lang]) {
    const got = tok.encode(text);
    if (got.length !== ids.length || got.some((x, i) => x !== ids[i])) {
      if (idMismatch++ < 3) console.log(`  ids differ: ${JSON.stringify(text.slice(0, 40))}\n    js ${got.slice(0, 12)}\n    py ${ids.slice(0, 12)}`);
    }
    if (tok.decode(ids) !== decoded) {
      if (textMismatch++ < 3) console.log(`  decode differs: ${JSON.stringify(tok.decode(ids).slice(0, 40))} vs ${JSON.stringify(decoded.slice(0, 40))}`);
    }
  }
  console.log(`${lang} tokenizer: ${fixture[lang].length} strings, ${idMismatch} id mismatches, ${textMismatch} decode mismatches`);
  failed ||= idMismatch > 0 || textMismatch > 0;

  const bank = read(`data/${lang}_questions.json`);
  for (const v of read("data/models.json")[lang]) {
    const reasoning = await load(`${lang}_${v.id}`);
    let ok = 0, n = 0, tokens = 0;
    const t0 = performance.now();
    for (const hops of [2, 3, 4, 5, 6, 8, 10]) {
      for (const q of bank.filter((x) => x.hops === hops).slice(0, 2)) {
        const added = await reasoning.generate(tok.encode(q.text), 80, tok.eosId, { temperature: 0 });
        tokens += added.length + 1;
        ok += assertedRelation(lang, splitRationale(tok.decode(added)).answer, q.query) === q.relation;
        n++;
      }
    }
    const secs = (performance.now() - t0) / 1000;
    console.log(`${lang} ${v.id}: ${ok}/${n} correct, ${(secs / n).toFixed(2)} s per answer (${(tokens / secs).toFixed(0)} tokens/s)`);
  }

  const lm = await load(`${lang}_lm`);
  const prompt = tok.encode(lang === "hindi" ? "भारत की राजधानी नई दिल्ली है और" : "অসমৰ ৰাজধানী গুৱাহাটী আৰু");
  const added = await lm.generate(prompt, 40, tok.eosId, { temperature: 0.8, topP: 0.9, repetitionPenalty: 1.15 });
  console.log(`${lang} generate: ${tok.decode([...prompt, ...added])}`);
}
process.exit(failed ? 1 : 0);
