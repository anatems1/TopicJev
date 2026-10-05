# Quick Start

This page is a short tour of the four modules. Each example downloads a small model from the
Hugging Face Hub on first use and runs on a laptop.

## Every backend is a context manager

All TopicJev backends share one lifecycle: the constructor only stores settings, `load_model()`
loads the weights onto the [selected device](devices.md), and `close()` frees them again. A `with`
block does both:

```python
from topicjev.entail import XEncoderEntail

model = XEncoderEntail("MoritzLaurer/deberta-v3-base-zeroshot-v2.0")  # nothing loaded yet
with model:                                                          # load_model()
    results = model.entail(docs=docs, lbls=labels)
# close(): weights moved off the GPU, accelerator cache emptied
```

This keeps memory under control when a pipeline uses several models one after another, for
example an embedder, then a summarizer, then a classifier.

## Classify documents

`entail()` scores every document against every label and returns one result per document:

```python
from topicjev.entail import XEncoderEntail

docs = [
    "NASA's Artemis program plans lunar base camp and rover deployments.",
    "Central banks consider interest rate cuts amid slowing inflation.",
]
labels = ["Space Exploration", "Monetary Policy", "Sports Organization"]

model = XEncoderEntail("MoritzLaurer/deberta-v3-base-zeroshot-v2.0")
with model:
    results = model.entail(docs=docs, lbls=labels)

for res in results:
    print(labels[res["class"]] if res["class"] < len(labels) else "Other", round(res["prob"], 4))
```

```text
Space Exploration 0.9997
Monetary Policy 0.9998
```

Every result is a dictionary:

```python
{
    "class": 0,         # index into labels; len(labels) means "Other"
    "prob": 0.9997,     # probability of the top-scoring candidate
    "other": None,      # name of the decoy label, if a decoy won
    "probs": [0.9997, 0.0001, 0.0002],  # one probability per label
}
```

Swap the class to change the scoring method; the call stays the same.
[Zero-shot Classification](entailment.md) compares all backends and explains decoys, probes and
thresholds.

## Compress long documents

Cross-encoders and small classifiers read only the first few hundred tokens of a document.
Compress long texts first:

```python
from topicjev.compress import CausalCompressor

compressor = CausalCompressor("Qwen/Qwen2.5-1.5B-Instruct", ratio=0.3)
with compressor:
    summaries = compressor.compress(long_docs)
```

`ratio=0.3` lets the summary use up to 30% of the prompt's length in tokens. See
[Compression](compression.md) for seq2seq summarizers and LLMLingua-2 token pruning.

## Embed documents

```python
from topicjev.gen import LocalEmbedder

embedder = LocalEmbedder("BAAI/bge-small-en-v1.5")
with embedder:
    vectors = embedder.embed_docs(["Document one content...", "Document two content..."])

print(vectors.shape)  # (2, 384)
```

Use `get_embeds()` with `save_dir=` to cache the embeddings of a whole corpus on disk; see
[Embeddings](embeddings.md).

## Generate structured output

`LocalGenerator` runs a prompt template over a batch of inputs and parses the answers as JSON,
retrying the ones that come back malformed:

```python
from topicjev.gen import LocalGenerator

gen = LocalGenerator("Qwen/Qwen2.5-1.5B-Instruct", temperature=0.0)
prompt = (
    "Extract key metadata from the document below as JSON with keys 'title' and 'keywords':\n\n"
    "TEXT:\n{text}\n\nJSON:"
)
inputs = [{"text": "CRISPR gene editing shows promising results in clinical trials for sickle cell disease."}]

with gen:
    results = gen.batch(prompt, inputs, req_keys=["title", "keywords"])
```

See [LLM Generation](generation.md), and [Prompt Templates](prompts.md) for the bundled prompts
that name and describe topics.

## Next steps

- The examples put the pieces together on a real corpus:
  [BERTopic + Laya](../examples/bertopic_laya.md) and
  [KMeans + LLM labels + Laya](../examples/kmeans_genlbs_laya.md).
- [How It Works](../algorithm.md) explains the idea behind the bootstrap.
