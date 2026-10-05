# Changelog

TopicJev is in early development: versions `0.x` may change the API, and `.devN` versions are test
releases.

## 0.1.0.dev3

*2026-10-05*

- **Repository moved** to [anatems1/TopicJev](https://github.com/anatems1/TopicJev).
- **New examples** in `examples/`, both on a 1,500-article slice of AG News loaded with
  Hugging Face `datasets` (replacing `bertopic-example.py` and the bundled PNAS data):
    - `bertopic_laya.py`: BERTopic topics, verified with `LayaEntail`;
    - `kmeans_genlbs_laya.py`: KMeans clusters, labeled by `Qwen3-1.7B` through the `reduce` →
      `classes` prompts, then verified with `LayaEntail`.
- The `examples` extra adds `datasets`.
- **README rewritten:** motivation, use cases, a road-feedback Quick Start, the examples, current
  limitations and acknowledgements; new logo.
- Docstrings cite LLMLingua-2 (`LinguaCompressor`) and GoalEx (`GoalExEntail`).

## 0.1.0.dev2

*2026-10-02*

- **NVIDIA GPUs on Windows and Linux:** `requirements-cuda.txt` installs PyTorch 2.14.1 built for
  CUDA 12.6 from the PyTorch package index, then TopicJev with all extras. The default Windows
  PyTorch from PyPI is CPU-only, so an NVIDIA GPU sat idle. The README explains how to check the
  driver with `nvidia-smi`.
- **Package metadata:** SPDX license `MIT` with `license-files`, authors, keywords and
  classifiers; requires hatchling 1.27 or newer.
- The version is read from `topicjev/__init__.py` only.
- `ruff` moved from a published `dev` extra to a `dev` dependency group.
- Ships `py.typed`, so type checkers use TopicJev's type hints.

## 0.1.0.dev1

*2026-10-02*

- **Fixed:** the wheel did not build (`python -m build` and `pip install .` failed; editable
  installs were unaffected). The bundled prompts were included twice.
- **The BERTopic example works on fresh clones, Apple Silicon and transformers 5:**
    - `nomic-embed-text-v1.5` is loaded through transformers' built-in NomicBert, because the
      checkpoint's own code fails on transformers 5. `LocalEmbedder` gained a `trust_remote_code`
      option (default `True`).
    - Settings are constants; results go to `output/<config hash>/` with a `config.json`, and the
      folder is created if missing. UMAP and KMeans are seeded from `RND_SEED`. Embeddings use
      nomic's `clustering: ` prefix.
- **Devices:** `detect_device()` prefers CUDA, then MPS, then the CPU, and `TOPICJEV_DEVICE`
  overrides it. MPS runs in `float32`.
- `empty_device_cache()` frees the CUDA or MPS cache; every `close()` uses it.
- Laya and LLMLingua keep the CUDA device index (e.g. `cuda:1`).
- Requires transformers 5.5 or newer and sentence-transformers 6.0 or newer.

## Initial release

*2026-10-01*

- `topicjev.compress`, `topicjev.entail` and `topicjev.gen` pipelines.
