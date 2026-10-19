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
short_description: 25M-param Hindi & Assamese LMs trained from scratch
---

# Hindi & Assamese mini language models

Small decoder-only Transformers trained from scratch for the *Language Models and Agents*
(Monsoon 2026) individual project
([code and reports](https://github.com/CL3-410/individual-project-Seronic2001)).

* **Generate**: the pretrained Modern V2 model of each language (RoPE, SwiGLU, RMSNorm;
  25.2M parameters; 500M training tokens) continues a prompt.
* **Reasoning**: the Baseline V1 model (learned positions, GELU; 25.0M parameters) after
  Direct fine-tuning on synthetic comparison questions answers a fresh question built from
  names it never saw in training. Training used chains of 1–3 hops; longer chains test
  generalisation (Hindi: 0.85 accuracy at 10 hops).

Both tokenizers are 16,384-piece SentencePiece BPE models trained on each language's corpus.
Everything runs on the free CPU hardware, with key/value caching during generation.
