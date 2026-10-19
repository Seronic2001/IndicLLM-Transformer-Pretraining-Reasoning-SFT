---
title: Hindi & Assamese mini-LMs
emoji: 📚
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: 25M-param Hindi & Assamese LMs running in the browser
---

# Hindi & Assamese mini language models

Small decoder-only Transformers trained from scratch for the *Language Models and Agents*
(Monsoon 2026) individual project
([code and reports](https://github.com/CL3-410/individual-project-Seronic2001)).
Everything runs in the visitor's browser with ONNX Runtime Web; `app.py` only serves the
page and the model files. The same page is also deployed on Railway.

* **Generate**: the pretrained Modern V2 model of each language (RoPE, SwiGLU, RMSNorm;
  25.2M parameters; 500M training tokens) continues a prompt.
* **Reasoning**: four fine-tunes per language (Baseline V1 or Modern V2, answering Direct or
  with a chain of thought) answer a comparison question built from names they never saw in
  training, checked against a symbolic solver, with each model's measured accuracy at the
  chosen depth. Training used chains of 1–3 hops; longer chains test generalisation.
* **Tokenizer**: the 16,384-piece SentencePiece BPE pieces of any text, re-implemented in
  JavaScript (`lm.js`).

Models are exported to ONNX with a key/value cache and fp16 weights (~64 MB each; each one
downloads the first time it is used).
