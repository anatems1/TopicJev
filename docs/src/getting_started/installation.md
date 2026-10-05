# Installation

## Requirements

- Python **3.12** or newer.
- PyTorch, Transformers and Sentence-Transformers. They are installed as dependencies.
  TopicJev is tested with Python 3.12, torch 2.14, transformers 5.18 and sentence-transformers 6.1.
- Optional: an NVIDIA GPU (CUDA) or an Apple Silicon Mac (MPS). Everything also runs on the CPU,
  only slower.

## Create a virtual environment

Use a fresh virtual environment, so TopicJev's pinned dependencies do not clash with other
projects:

=== "macOS / Linux"

    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    ```

=== "Windows (PowerShell)"

    ```powershell
    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    ```

## Install from source

```bash
git clone https://github.com/anatems1/TopicJev.git
cd TopicJev
pip install -e .
```

The core install contains every backend except LLMLingua-2. Optional extras add more:

| Extra | Adds | Needed for |
| :--- | :--- | :--- |
| `lingua` | `llmlingua` | [`LinguaCompressor`](compression.md#linguacompressor) |
| `examples` | `bertopic`, `pandas`, `scikit-learn`, `datasets` | The [examples](../examples/bertopic_laya.md) |
| `all` | All of the above | Everything |

```bash
pip install -e ".[lingua,examples]"   # pick extras
pip install -e ".[all]"               # or install everything
```

## GPU support

TopicJev uses whatever device the installed PyTorch build can reach: CUDA first, then Apple
Silicon (MPS), then the CPU. See [Hardware and Memory](devices.md) for details and for the
`TOPICJEV_DEVICE` override.

### NVIDIA GPU (Windows, Linux)

On Windows, a plain `pip install` gets a CPU-only build of PyTorch from PyPI, so an NVIDIA GPU sits
idle. Install the CUDA build instead, from the repository root:

```bash
pip install -r requirements-cuda.txt
```

This installs `torch==2.14.1+cu126` from the PyTorch package index (on Windows x64 and
Linux x86_64), then TopicJev itself with all extras.

!!! warning "Check your CUDA driver first"

    Run `nvidia-smi`: the `CUDA Version` in its header must be **12.6 or higher**, because
    `requirements-cuda.txt` installs PyTorch built for CUDA 12.6. With an older driver, PyTorch
    installs but cannot use the GPU, and everything silently runs on the CPU. Update the driver,
    or pick a matching build with the [PyTorch selector](https://pytorch.org/get-started/locally/).

Verify that PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.14.1+cu126 True
```

### Apple Silicon (macOS)

Nothing extra is needed: the PyPI build of PyTorch supports Apple Silicon GPUs (MPS), and TopicJev
uses them automatically.

## Verify the installation

```bash
python -c "import topicjev; print(topicjev.__version__)"
python -c "from topicjev.backend import detect_device; print(detect_device())"
```

The second command prints the device and precision TopicJev will use, for example
`(device(type='mps'), torch.float32)` on an Apple Silicon Mac.

## Models and credentials

TopicJev does not ship model weights. Every backend takes a
[Hugging Face Hub](https://huggingface.co/models) model name (or a local path) and downloads the
weights on first use into the Hugging Face cache (`~/.cache/huggingface/hub` by default; set
`HF_HOME` to move it). Later runs reuse the cache.

- **Gated models** such as `meta-llama/*` need an accepted license on the Hub and a login:
  `hf auth login`.
- **Offline runs**: once the models are cached, set `HF_HUB_OFFLINE=1` to skip all network calls.
- **TypeSafe API**: [`JevEntail`](entailment.md#layaentail-and-jeventail) reads its key from the
  `TYPESAFE_API_KEY` environment variable, or from a `.env` file in the working directory, e.g.
  at the repository root:

    ```bash
    TYPESAFE_API_KEY="your-api-key-here"
    ```

## Development tools

Development tools are declared as [dependency groups](https://packaging.python.org/en/latest/specifications/dependency-groups/),
which need pip 25.1 or newer:

```bash
pip install --group dev    # ruff
pip install --group docs   # MkDocs, to build this documentation
```
