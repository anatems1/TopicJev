# Frequently Asked Questions

## Classification

### Why are almost all my documents assigned to *Other*?

Check these, in order:

1. **You have one or two labels.** The default threshold is \(2/N\): for two labels that is `1.0`,
   which no probability exceeds, so every document becomes *Other*. Pass `threshold=0.5`, or use
   `multi_lbl=True` for a single label. See [Threshold](getting_started/entailment.md#threshold).
2. **A decoy wins.** Look at the `other` field of the results: it names the decoy that took the
   document. With `LayaEntail` and `JevEntail`, a built-in *Other* option is always present.
3. **The labels are too thin.** Five keywords may not tell the classifier enough. Write a title
   and a one-sentence description per topic, for example with the
   [bundled prompts](getting_started/prompts.md).
4. **The documents are long.** Classifiers see only the first few hundred tokens; the rest is
   truncated. [Compress](getting_started/compression.md) the documents first.

### What does `class == len(labels)` mean?

*Other*: no label passed the threshold, or a decoy scored highest. Map results to names with:

```python
names = [labels[r["class"]] if r["class"] < len(labels) else "Other" for r in results]
```

### Why do the `probs` not sum to 1?

`probs` lists only your labels. Decoys, including the built-in *Other* option of `LayaEntail` and
`JevEntail`, take part in the softmax but are left out of `probs`. In
[multi-label mode](getting_started/entailment.md#multi-label-mode), each probability is independent
and the sum can be anything.

### Why is every probability exactly 0 or 1?

`GoalExEntail` and `CausalEntail` default to `temperature=0.1`, which sharpens the scores a lot.
Pass `temperature=1.0` if you need graded confidences.

### Can a document get more than one label?

Pass `multi_lbl=True` and read `probs`; `class` still holds only the best label. See
[Multi-label mode](getting_started/entailment.md#multi-label-mode). `LayaEntail` and `JevEntail`
are single-label only.

## The examples

### An example reuses old results after I changed the script

The [examples](examples/bertopic_laya.md) cache their results in `output/<hash>/`, where the hash
covers only the settings in `CONFIG`. If you change anything else, such as the BERTopic parameters
or a prompt, add it to `CONFIG` or delete the output folder.

### `'NomicBertModel' object has no attribute 'get_extended_attention_mask'`

The modeling code that ships with `nomic-ai/nomic-embed-text-v1.5` was written for transformers 4
and fails on transformers 5. Load the model with transformers' built-in implementation instead:

```python
LocalEmbedder("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=False)
```

This needs transformers 5.5 or newer, which TopicJev already requires.

### `This modeling file requires the following packages that were not found in your environment: einops`

The same remote modeling code needs `einops`. Installing it only leads to the error above; use
`trust_remote_code=False` instead.

## Installation and hardware

### I have an NVIDIA GPU, but everything runs on the CPU

PyTorch from PyPI is CPU-only on Windows, and TopicJev falls back to the CPU without a warning.
Install the CUDA build with `pip install -r requirements-cuda.txt`, after checking that
`nvidia-smi` reports CUDA 12.6 or higher. See
[NVIDIA GPU](getting_started/installation.md#nvidia-gpu-windows-linux).

### How do I force the CPU, or pick a GPU?

Set `TOPICJEV_DEVICE`, for example `TOPICJEV_DEVICE=cpu` or `TOPICJEV_DEVICE=cuda:1`. See
[Hardware and Memory](getting_started/devices.md#choosing-a-device-yourself).

### `Segmentation fault: 11` or `OMP: Error #15` on macOS

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

FAISS and PyTorch each bundle their own copy of the OpenMP runtime on macOS, and a multi-threaded
FAISS search crashes when both are loaded. This hits code that calls `index.search()` on the
embedder's FAISS index, such as the
[KMeans + LLM labels example](examples/kmeans_genlbs_laya.md). Run with a single OpenMP thread:

```bash
OMP_NUM_THREADS=1 python examples/kmeans_genlbs_laya.py
```

Models on the GPU (MPS) are not affected; only CPU-side OpenMP work runs single-threaded.

### Installing BERTopic fails while building `hdbscan`

On some platforms, Linux on ARM for example, `hdbscan` (a BERTopic dependency) has no prebuilt
wheel for Python 3.12 and is compiled from source. Install a C compiler first (`build-essential`
on Debian and Ubuntu; the Microsoft C++ Build Tools on Windows) and retry.

### Where are the models stored, and can I run offline?

Models are downloaded on first use into the Hugging Face cache, `~/.cache/huggingface/hub` by
default (set `HF_HOME` to move it). Once they are cached, set `HF_HUB_OFFLINE=1` to run without
network access.

### I run out of GPU memory

Lower `batch_size`, run one model at a time in its own `with` block, or use smaller checkpoints. See
[Running out of memory](getting_started/devices.md#running-out-of-memory).

## Embeddings

### My embeddings do not match my documents

`LocalEmbedder.get_embeds()` loads `<save_dir>/embeds.bin` whenever it exists, without checking
whether the documents or the model have changed. Use a separate `save_dir` for every corpus and
model, or delete the file. See [Embeddings](getting_started/embeddings.md#embed-a-corpus-with-a-cache).
