# `topicjev.entail`

Zero-shot and entailment-based document classification backends for topic assignment and verification.

---

## Architecture Overview

All backends inherit from the core lifecycle abstract base class `Entailment` (which subclasses `ModelBackend`), providing unified model resource management (`load_model()`, `close()`, and context manager support `with model:`).

```
ModelBackend
    │
    ├── LocalModelBackend
    │    └── CausalLMBackend (+ ChatTemplateMixin)
    │         └── CausalEntail (Subclasses CausalLMBackend + GoalExEntail)
    │
    └── Entailment (ABC)
         ├── LocalEntailment (Base for PyTorch/CUDA local models)
         │    ├── Seq2SeqEntail (Teacher-forced target log-likelihood)
         │    │    └── GoalExEntail (+ YesNoLogitMixin: First decoder step Yes/No logit gap)
         │    │         └── CausalEntail (Terminal token Yes/No logit gap)
         │    └── XEncoderEntail (NLI Premise-Hypothesis cross-encoder)
         └── RemoteEntailment (Base for hosted API clients)
              └── JevEntail (TypeSafe System-One decision endpoints)
                   └── LayaEntail (Local Laya decision agent)
```

---

## Supported Backends

| Backend Class | Model Family | Scoring Method | Default Temp | Multi-Label Support |
| :--- | :--- | :--- | :--- | :--- |
| `Seq2SeqEntail` | Encoder-Decoder (e.g. FLAN-T5) | Mean teacher-forced target token log-likelihood | `1.0` | Softmax / Sigmoid |
| `GoalExEntail` | Encoder-Decoder (e.g. FLAN-T5) | Initial decoder token binary logit gap ($\log P(\text{yes}) - \log P(\text{no})$) | `0.1` | Softmax / Sigmoid |
| `CausalEntail` | Causal LM (e.g. Llama, Qwen) | Terminal token binary logit gap ($\log P(\text{yes}) - \log P(\text{no})$) | `0.1` | Softmax / Sigmoid |
| `XEncoderEntail` | Cross-Encoder (e.g. DeBERTa-v3 NLI) | NLI entailment logit space (binary / ternary / raw) | `1.0` | Softmax / Sigmoid |
| `JevEntail` | Hosted TypeSafe API | System-One multi-criteria classification | `1.0` | Softmax |
| `LayaEntail` | Local Laya Agent | Local agent multi-criteria classification | `1.0` | Softmax |

---

## Scoring & Calibration

1. **Decoys**: Distractor labels (`decoys=[...]`) compete for probability mass to prevent out-of-domain documents from falsely receiving high confidence under a closed taxonomy.
2. **Contextual Probes**: Providing probe baseline documents (`probes=[...]`) subtracts null-context model bias before temperature scaling.
3. **Temperature Scaling**: `scores = scores / temperature` sharpens or smooths confidence before softmax/sigmoid.
4. **Confidence Thresholding**: Default threshold is $2 / N_{\text{labels}}$ for single-label (or $0.5$ for multi-label). Documents failing the threshold or matching decoys are assigned to class $N_{\text{labels}}$ (`Other`).

---

## Quickstart

```python
from topicjev.entail import CausalEntail

docs = [
    "NASA's Artemis program plans lunar base camp and rover deployments.",
    "Central banks consider interest rate cuts amid slowing inflation.",
]
labels = ["Space Exploration", "Monetary Policy", "Sports Organization"]

model = CausalEntail("meta-llama/Llama-3.2-3B-Instruct")
with model:
    results = model.entail(docs=docs, lbls=labels)

for doc, res in zip(docs, results):
    print(f"Assigned Class {res['class']}: prob={res['prob']:.4f}, other={res['other']}")
```

### Result Schema

```python
{
    "class": 0,                # Integer label index (or len(labels) for 'Other')
    "prob": 0.9412,            # Top probability score
    "other": None,             # Label name if assigned to decoy / out-of-taxonomy
    "probs": [0.9412, 0.05, 0.0088]  # Probabilities across taxonomy labels
}
```
