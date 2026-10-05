# Embeddings

`LocalEmbedder` turns documents into dense vectors with any
[Sentence-Transformers](https://sbert.net) model. The vectors feed a topic model such as BERTopic,
and can be cached on disk so a corpus is embedded only once.

## Embed a few documents

```python
from topicjev.gen import LocalEmbedder

embedder = LocalEmbedder("BAAI/bge-small-en-v1.5", batch_size=64, normalize_embeddings=True)
texts = ["Document one content...", "Document two content..."]

with embedder:
    vectors = embedder.embed_docs(texts)

print(vectors.shape)  # (2, 384)
```

`embed_docs()` returns a NumPy array of shape `(n_docs, dim)`. It loads the model on first use, so
the `with` block is optional; it only makes sure the model is freed afterwards.

## Embed a corpus, with a cache

`get_embeds()` embeds a whole corpus in chunks of `batch_size` documents, with a progress bar. With
`save_dir`, it also stores the vectors in a [FAISS](https://github.com/facebookresearch/faiss) index
at `<save_dir>/embeds.bin`, and on the next call loads them from there without loading the model:

```python
embedder = LocalEmbedder("BAAI/bge-small-en-v1.5", save_dir="output/run-1")
embeds = embedder.get_embeds(docs)   # first run: embeds and writes output/run-1/embeds.bin
embeds = embedder.get_embeds(docs)   # later runs: read from disk
```

!!! warning "The cache does not know what it contains"
    `get_embeds()` reuses `embeds.bin` whenever the file exists. It does not check whether the
    documents, the model or the prefix have changed since. Use a separate `save_dir` per corpus
    and model, as the [examples](../examples/bertopic_laya.md#1-settings-and-the-output-folder) do by
    naming its output folder after a hash of its settings.

The result plugs straight into BERTopic:

```python
topics, probs = topic_model.fit_transform(docs, embeddings=embeds)
```

## Task prefixes

Many retrieval embedders are trained with a prefix that tells the model what the text is for.
`prefix` is prepended to every document:

```python
embedder = LocalEmbedder(
    "nomic-ai/nomic-embed-text-v1.5",
    prefix="clustering: ",
    trust_remote_code=False,
)
```

| Model family | Prefix for documents to cluster |
| :--- | :--- |
| `nomic-ai/nomic-embed-text-v1.5` | `"clustering: "` |
| `intfloat/e5-*` | `"query: "` (E5 uses it for symmetric tasks such as clustering) |
| BGE, MiniLM, most Sentence-Transformers models | none |

Check the model card for the exact prefix; a wrong or missing prefix quietly degrades the vectors.

## Parameters

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `batch_size` | `128` | Documents per chunk in `get_embeds()`. |
| `prefix` | `""` | Text prepended to every document. |
| `normalize_embeddings` | `True` | L2-normalize the vectors, so a dot product equals cosine similarity. |
| `trust_remote_code` | `True` | Allow the checkpoint's own modeling code. Set `False` to use the implementation built into transformers, if it has one. |
| `save_dir` | `None` | Folder for the `embeds.bin` cache used by `get_embeds()`. |

The model runs on the [detected device](devices.md) in its default precision: `bfloat16` or
`float16` on CUDA, `float32` elsewhere.

!!! tip "`trust_remote_code` and transformers 5"
    Some checkpoints ship modeling code written for older transformers releases. `nomic-embed-text-v1.5`
    is one: its remote code fails on transformers 5, while the `NomicBert` implementation built into
    transformers (since 5.5) works. Pass `trust_remote_code=False` for such models. See the
    [FAQ](../faq.md#nomicbertmodel-object-has-no-attribute-get_extended_attention_mask).
