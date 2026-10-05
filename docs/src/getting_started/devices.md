# Hardware and Memory

## Device selection

Every local backend picks its device when it is constructed, with
[`detect_device()`](../api/backend.md#topicjev.backend.detect_device):

1. **CUDA**, if PyTorch can see an NVIDIA (or AMD ROCm) GPU;
2. **MPS**, on an Apple Silicon Mac;
3. the **CPU** otherwise.

The precision follows the device:

| Device | Precision |
| :--- | :--- |
| CUDA, compute capability 8.0 or newer (Ampere and later) | `bfloat16` |
| Older CUDA GPUs | `float16` |
| MPS | `float32`, so scores match CPU runs |
| CPU | `float32` |

Check what TopicJev will use:

```bash
python -c "from topicjev.backend import detect_device; print(detect_device())"
```

## Choosing a device yourself

Set `TOPICJEV_DEVICE` to any PyTorch device string to override the detection:

=== "macOS / Linux"

    ```bash
    TOPICJEV_DEVICE=cpu python examples/bertopic_laya.py      # force the CPU
    TOPICJEV_DEVICE=cuda:1 python examples/bertopic_laya.py   # second GPU
    ```

=== "Windows (PowerShell)"

    ```powershell
    $env:TOPICJEV_DEVICE = "cpu"
    python examples/bertopic_laya.py
    ```

The variable is read when a backend is created, so it can also be set from Python before the
backends are constructed:

```python
import os
os.environ["TOPICJEV_DEVICE"] = "cpu"
```

Two backends deviate from the rule: `LinguaCompressor` runs on the CPU unless the device is CUDA,
so it uses the CPU on Apple Silicon too, and `JevEntail` runs remotely and uses no local device.

!!! warning "An NVIDIA GPU, but everything runs on the CPU?"
    TopicJev can only use what the installed PyTorch build supports, and it falls back to the CPU
    without a warning. On Windows, the default PyTorch from PyPI is CPU-only. See
    [NVIDIA GPU](installation.md#nvidia-gpu-windows-linux) to install the CUDA build.

## Memory lifecycle

Model weights are only loaded when you enter a `with` block (or call `load_model()`), and released
when you leave it (or call `close()`). On close, local backends:

1. move the model to the CPU,
2. drop their references to the model and tokenizer,
3. empty the CUDA or MPS cache.

A pipeline that uses several large models should therefore run them one after another, each in its
own block:

```python
with LocalEmbedder("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=False) as embedder:
    embeds = embedder.get_embeds(docs)

with CausalCompressor("Qwen/Qwen2.5-1.5B-Instruct") as compressor:
    short_docs = compressor.compress(docs)

with XEncoderEntail("MoritzLaurer/deberta-v3-base-zeroshot-v2.0") as model:
    results = model.entail(docs=short_docs, lbls=labels)
```

Only one model is in GPU memory at any time.

## Running out of memory

- Lower `batch_size`. For local entailment backends it counts (document, label) pairs, so 100
  documents and 10 labels make 1,000 pairs.
- Close models you no longer need, or use the `with` pattern above.
- Use a smaller checkpoint, or [compress](compression.md) long documents first: shorter inputs need
  less memory per pair.
