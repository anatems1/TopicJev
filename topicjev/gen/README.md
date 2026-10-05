# `topicjev.gen`

Text generation and dense representation backends for topic labeling, cluster synthesis, and document embeddings.

---

## Architecture Overview

All backends inherit from `ModelBackend`, providing unified model lifecycle management (`load_model()`, `close()`, and `with backend:`).

```
ModelBackend
    │
    ├── LocalModelBackend
    │    └── CausalLMBackend (+ ChatTemplateMixin)
    │         └── LocalGenerator (Causal LM text/JSON generation)
    │
    ├── Generator (ABC)
    │    └── LocalGenerator (Subclasses Generator + CausalLMBackend)
    │
    └── Embedder (ABC)
         └── LocalEmbedder (SentenceTransformers dense embeddings)
```

---

## Supported Backends

### 1. `LocalGenerator`

Inference engine for local causal language models (e.g. Llama, Mistral, Qwen):
- **Chat Template Wrapping**: Automatically formats user and system messages using tokenizer chat templates (with graceful fallback for reasoning models).
- **Structured JSON Mode**: Strips markdown code blocks and validates outputs against required schema keys (`req_keys=[...]`).
- **Resilient Retry Chain**: Retries malformed JSON generations up to `max_retries` times per document.
- **Pipeline Interoperability**: Exposes both `.batch()` and `.run_chain()` methods.

### 2. `LocalEmbedder`

Dense vector embedding engine powered by `sentence-transformers`:
- **GPU Acceleration**: Uses device and precision auto-detection (`bfloat16`/`float16`).
- **Batch Processing**: Configurable batch sizing and L2 embedding normalization.
- **Prefix Support**: Supports query/passage instruction prefixes required by modern bi-encoders (e.g. E5, BGE).

---

## Quickstart

### Structured Generation with `LocalGenerator`

```python
from topicjev.gen import LocalGenerator

gen = LocalGenerator(
    mname="meta-llama/Llama-3.2-3B-Instruct",
    json_mode=True,
    temperature=0.0,
)

prompt_template = (
    "Extract key metadata from the document below as JSON with keys 'title' and 'keywords':\n\n"
    "TEXT:\n{text}\n\nJSON:"
)

docs = [
    {"text": "CRISPR gene editing shows promising results in clinical trials for sickle cell disease."},
    {"text": "Quantum computing algorithms achieve polynomial speedups for specific simulation tasks."},
]

with gen:
    results = gen.batch(prompt_template, docs, req_keys=["title", "keywords"])

for item in results:
    print(item["title"], item["keywords"])
```

### Document Embeddings with `LocalEmbedder`

```python
from topicjev.gen import LocalEmbedder

embedder = LocalEmbedder(
    mname="BAAI/bge-small-en-v1.5",
    batch_size=64,
    normalize_embeddings=True,
)

texts = ["Document one content...", "Document two content..."]

with embedder:
    vectors = embedder.embed_docs(texts)

print(f"Computed embeddings matrix of shape: {vectors.shape}")
```
