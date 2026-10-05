# `topicjev.compress`

Document and prompt compression backends for condensing long-form texts into compact representations prior to downstream clustering or classification.

---

## Architecture Overview

All compression backends subclass `Compressor` (which inherits `ModelBackend`), providing standard hardware device detection and resource lifecycle management (`load_model()`, `close()`, and `with compressor:`).

```
ModelBackend
    │
    ├── LocalModelBackend
    │    └── CausalLMBackend (+ ChatTemplateMixin)
    │         └── CausalCompressor (Input-relative dynamic causal LM summarization)
    │
    └── Compressor (ABC)
         ├── CNNCompressor (Seq2seq fixed-budget summarization, e.g. BART, PEGASUS)
         │    └── Seq2SeqCompressor (Instruction-guided seq2seq summarization, e.g. FLAN-T5)
         ├── CausalCompressor (Subclasses Compressor + CausalLMBackend)
         └── LinguaCompressor (LLMLingua2 token-level pruning without generation)
```

---

## Supported Backends

| Backend Class | Mechanism | Budget Formulation | Ideal Use Case |
| :--- | :--- | :--- | :--- |
| `CNNCompressor` | Seq2Seq generation | `ratio * model_context_len` | Checkpoints fine-tuned for fixed-length outputs (e.g. CNN/DailyMail). |
| `Seq2SeqCompressor` | Seq2Seq generation + prompt | `ratio * model_context_len` | General instruction-following encoder-decoder models (e.g. FLAN-T5). |
| `CausalCompressor` | Causal LM generation | `ratio * input_prompt_len` | Modern causal chat models (Llama, Qwen, Mistral) where budget should scale with document length. |
| `LinguaCompressor` | Token pruning (LLMLingua2) | `ratio` retention rate | High-throughput extractive token dropping preserving verbatim key tokens. |

---

## Quickstart

### Causal LM Summarization

```python
from topicjev.compress import CausalCompressor

compressor = CausalCompressor(
    mname="meta-llama/Llama-3.2-3B-Instruct",
    ratio=0.3,
    batch_size=4,
)

docs = [
    "Extensive document describing climate models, carbon emissions, and climate mitigation strategies...",
    "Detailed report discussing municipal transit systems, bus electrification, and ridership trends...",
]

with compressor:
    summaries = compressor.compress(docs)

for summary in summaries:
    print(summary)
```

### Extractive Token Pruning

```python
from topicjev.compress import LinguaCompressor

compressor = LinguaCompressor(
    mname="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    ratio=0.5,
)

with compressor:
    pruned_docs = compressor.compress(docs)
```
