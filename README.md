[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# TopicJev

<p align="center">
  <img src="https://github.com/anatems1/TopicJev/blob/master/images/topicjev.png" width="40%" alt="TopicJev Logo" />
</p>

**An Efficient Low-Resource Bootstrap for Topic Model Refinement in Zero-Shot Settings**

Dense embedding models and clustering algorithms for topic modeling tend to group documents by similarity or geometry. As clusters grow, they inevitably absorb documents that are only superficially related. Without ground-truth labels, validating these groupings requires expensive manual audits or costly calls to hosted LLMs.

**TopicJev** can be employed to assess and refine your topic model alignment: it expresses each topic as a label (either its keywords or a short generated claim such as *"The text discusses…"*) and runs a zero-shot entailment model against the derived schema. The bootstrap nature of this pipeline allows virtually any topic model output (such as [BERTopic](https://github.com/MaartenGr/BERTopic) or k-means) to be cross-checked; hence its *refining* capability. Documents that support the label form the topic's refined core; the rest are flagged as outliers or routed to "Other" rather than silently diluting the topic.


<p align="center">
  <img src="https://github.com/anatems1/TopicJev/blob/master/images/refine_topics.png" width="100%" alt="Topic refinement abstract view." />
</p>

The new era of classification models, termed "System One" from [TypeSafe AI](https://typesafe.ai/), has sparked interest in the developer community for their low costs and fast inferencing. Prior to [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), zero-shot entailment methods (e.g., NLI with "entail", "neutral", and "contradict" classes) served as the foundation for hypothesis-driven classification. Today, open-source models (such as [Laya](https://github.com/NandhaKishorM/laya)) have surfaced to mirror the capabilities and intent of System One architectures. The purpose of **TopicJev** is to ultimately leverage the capabilities of foundational models and their modern lightweight counterparts to enhance representation, verification, and alignment in existing topic modeling frameworks on low-resource machines.

## Why TopicJev?

TopicJev is designed for scenarios where cluster quality matters but labeled data and massive compute budgets are absent:

- **Auditing & Refining Topic Models:** Post-process outputs from existing frameworks to purge false positives and calculate agreement metrics without ground-truth labels.
- **Zero-Shot Taxonomy Routing:** Classify unannotated corpora into user-defined categories with automatic gating to an `"Other"` class when confidence fails threshold criteria.
- **Low-Resource & Privacy-Preserving:** Operates locally on consumer GPUs, Apple Silicon (MPS), or CPU using compact models (e.g., ModernBERT decision encoders, cross-encoders, or small language models) - eliminating external API fees and rate limits.
- **Handling Long Documents:** Includes native prompt and document compression (LLMLingua-2 token pruning or causal summarization) to fit long texts into small model context windows.

## Core Modules

| Module | Purpose | Key Implementations |
| :--- | :--- | :--- |
| [`topicjev.compress`](https://github.com/anatems1/TopicJev/blob/master/compress/README.md) | Condenses long texts prior to downstream clustering or verification. | `LinguaCompressor` (extractive token pruning), `CausalCompressor` (length-scaled summarization), `Seq2SeqCompressor`. |
| [`topicjev.gen`](https://github.com/anatems1/TopicJev/blob/master/gen/README.md) | Dense embeddings and structured LLM generation. | `LocalEmbedder` (SentenceTransformers + FAISS indexing/caching), `LocalGenerator` (schema-constrained JSON extraction with auto-retries). |
| [`topicjev.entail`](https://github.com/anatems1/TopicJev/blob/master/entail/README.md) | Calibrated zero-shot entailment and topic verification. | `LayaEntail` (local ModernBERT decision model), `CausalEntail` (terminal logit gap), `XEncoderEntail` (cross-encoder NLI), `JevEntail` (hosted API). |

## Installation

```bash
# Basic installation
pip install -e .

# With optional backends
pip install -e ".[lingua,examples]"

# Or install all extras (including example dependencies)
pip install -e ".[all]"
```

Tested with Python 3.12, torch 2.14, transformers 5.18, and sentence-transformers 6.1.

### Hardware Acceleration

- **Apple Silicon (macOS):** MPS acceleration is detected and utilized automatically.
- **NVIDIA CUDA (Windows, Linux):** PyPI defaults to a CPU-only build of torch on Windows. Install the CUDA 12.6 build from the repository root:

```bash
pip install -r requirements-cuda.txt
```

> **Check your CUDA driver first.** Run `nvidia-smi`: the `CUDA Version` in its header must be
> **12.6 or higher**, because `requirements-cuda.txt` installs torch built for CUDA 12.6. With an
> older driver, torch installs but cannot use the GPU, and everything silently runs on the CPU.
> Update the driver, or pick a matching build with the
> [PyTorch selector](https://pytorch.org/get-started/locally/).

Verify that torch sees the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.14.1+cu126 True
```

### API Keys & Configuration

If you plan to use hosted decision backends (such as the TypeSafe AI endpoint via `JevEntail`), configure your API credentials in a `.env` file at the repository root:

```bash
TYPESAFE_API_KEY="your-api-key-here"
```

## Quick Start

Check documents against topic keywords (or generated claims) discovered by your topic model and route outliers to `"Other"`:

```python
import json
from topicjev.entail import LayaEntail

# Public feedback on roadway conditions
docs = [
    "Huge deep pothole on the right lane of I-80 eastbound near exit 15, nearly blew my tire out.",
    "Oak Street is closed off to all thru traffic between Maple and Pine for emergency bridge repair.",
    "Massive congestion on inbound Parkway East this morning, stop-and-go delays for over 4 miles.",
    "The new LED streetlights on Elm Street are too dim and flickering at night.",  # Outlier
]

# Simulated topic model results (topic name -> discovered keywords)
topics = {
    "pothole": "pothole, crater, asphalt damage, rim damage, blown tire, pavement defect",
    "road closure": "road closure, detour, closed, barricades, impassable, blocked, repair",
    "congestion": "congestion, traffic, gridlock, bottleneck, delays, stop and go, backup",
}

# Encode topics into format for Jev decision criteria (JSON)
topic_labels = [json.dumps({topic: desc}) for topic, desc in topics.items()]
topic_names = list(topics.keys())

# Initialize entailment model (Laya ModernBERT decision encoder)
model = LayaEntail("convaiinnovations/laya-typed-decisions")
with model:
    results = model.entail(docs=docs, lbls=topic_labels)

# Example output for results[0]:
# {
#   "class": 0,
#   "prob": 0.7239,
#   "other": None,
#   "probs": [0.7239, 0.0812, 0.0750]
# }

for doc, res in zip(docs, results):
    assigned = res["class"]
    prob = res["prob"]
    if assigned < len(topic_names):
        print(f"[{topic_names[assigned].upper():<12}] (prob: {prob:.2f}) -> '{doc[:55]}...'")
    else:
        # Out-of-domain documents are automatically gated to 'Other' rather than diluting clusters
        print(f"[{'OUTLIER/OTHER':<12}] (prob: {prob:.2f}) -> '{doc[:55]}...'")
```

**Expected Console Output:**
```text
[POTHOLE     ] (prob: 0.72) -> 'Huge deep pothole on the right lane of I-80 eastbound n...'
[ROAD CLOSURE] (prob: 0.75) -> 'Oak Street is closed off to all thru traffic between Ma...'
[CONGESTION  ] (prob: 0.91) -> 'Massive congestion on inbound Parkway East this morning...'
[OUTLIER/OTHER] (prob: 0.65) -> 'The new LED streetlights on Elm Street are too dim and ...'
```

**End-to-End Examples:**

For complete, reproducible workflows running on a 1,500-sample slice of [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), see the [`examples/`](examples/) directory:

- **[`bertopic_laya.py`](examples/bertopic_laya.py):** Integrates TopicJev with BERTopic (UMAP + KMeans + c-TF-IDF). It takes discovered keyword representations, structures them into decision criteria, and uses `LayaEntail` to verify document assignments and calculate topic agreement rates.
- **[`kmeans_genlbs_laya.py`](examples/kmeans_genlbs_laya.py):** A lightweight clustering pipeline that operates independently of BERTopic. It clusters dense embeddings using only KMeans, retrieves the top 5 centroid-nearest documents from the FAISS index, and chains local `Qwen3-1.7B` generation through the `reduce` $\to$ `classes` prompt pipeline to synthesize topic schemas before running `LayaEntail`.

## Current Limitations & Research Status

TopicJev is an active research project and work in progress. Please note the following current limitations:

- **Zero-Shot Confidence Calibration:** For local decision models (e.g., Laya) and the hosted TypeSafe Jev model, confidence calibration has not been fine-tuned. Topic assignments are truly zero-shot and determined by the argmax category selection and decoy routing; confidence scores should be treated as uncalibrated indicators rather than calibrated posterior probabilities.
- **Hosted API Batch Scaling:** High-volume batching of large document collections has not yet been evaluated with the remote TypeSafe API endpoint. We recommend experimenting with local models first before moving to the hosted endpoint for comparative evaluations.
- **Ongoing Benchmarks:** Formal empirical evaluations and quantitative benchmarks across standard topic modeling datasets are actively underway and will be reported in an upcoming release.

## Acknowledgements

TopicJev builds upon and integrates ideas from:

- [TypeSafe AI & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) for System One decision formulations.
- [Laya](https://github.com/NandhaKishorM/laya) for open-source ModernBERT decision architectures.
- [GoalEx](https://arxiv.org/abs/2305.13749) for target token logit extraction.
- [LLMLingua](https://github.com/microsoft/LLMLingua) for prompt and document token compression.


## License

[MIT](https://opensource.org/license/mit)