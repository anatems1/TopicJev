# Zero-shot Classification

`topicjev.entail` decides which of a set of candidate labels a document belongs to, without any
training data. It is the core of the TopicJev bootstrap: the labels are the topics of a topic model,
and the classifier checks whether each document really fits its topic.

All backends share one method:

```python
results = model.entail(docs=docs, lbls=labels)
```

and differ only in *how* they score a (document, label) pair.

## Results

`entail()` returns one dictionary per document, in input order:

| Key | Type | Meaning |
| :--- | :--- | :--- |
| `class` | `int` | Index of the assigned label in `lbls`. `len(lbls)` means *Other*: no label passed the [threshold](#threshold), or a [decoy](#decoys) won. |
| `prob` | `float` | Probability of the top-scoring candidate, whether a label or a decoy. |
| `other` | `str` or `None` | The decoy's text when a decoy scored highest, otherwise `None`. |
| `probs` | `list[float]` | One probability per label in `lbls`, in the same order. Decoys are left out, so with decoys the values can sum to less than 1. |

To turn results into label names:

```python
names = [labels[r["class"]] if r["class"] < len(labels) else "Other" for r in results]
```

## Choosing a backend

| Backend | Model family | Score of a (document, label) pair | Default temperature | Multi-label |
| :--- | :--- | :--- | :--- | :--- |
| [`XEncoderEntail`](#xencoderentail) | NLI cross-encoder (e.g. DeBERTa-v3) | Entailment logit of "The text discusses *label*" | `1.0` | Yes |
| [`GoalExEntail`](#goalexentail) | Encoder-decoder (e.g. FLAN-T5) | \(\log P(\text{yes}) - \log P(\text{no})\) at the first decoder step | `0.1` | Yes |
| [`CausalEntail`](#causalentail) | Causal chat LM (e.g. Qwen, Llama) | \(\log P(\text{yes}) - \log P(\text{no})\) at the next token | `0.1` | Yes |
| [`Seq2SeqEntail`](#seq2seqentail) | Encoder-decoder (e.g. FLAN-T5) | Mean log-likelihood of the label's tokens as the answer | `1.0` | Yes |
| [`LayaEntail`](#layaentail-and-jeventail) | Local Laya decision model (ModernBERT) | Laya's choice probabilities | `1.0` | No |
| [`JevEntail`](#layaentail-and-jeventail) | Hosted TypeSafe API | TypeSafe System-1 choice probabilities | `1.0` | No |

As a rule of thumb:

- Start with **`XEncoderEntail`** or **`LayaEntail`**: they are small, fast and made for this task.
- Use **`GoalExEntail`** or **`CausalEntail`** when a label is a longer property or claim that
  benefits from an instruction-following model.
- Local backends score every document against every label, so the cost grows with
  `len(docs) * len(labels)`. Laya and Jev score all labels of a document in one call.

## Backends

### XEncoderEntail

Uses a Natural Language Inference cross-encoder. Each document is the *premise*; each label is
turned into a *hypothesis* with a template, by default `"The text discusses {}"`.

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
```

```text
# results, rounded to 4 digits
[{'class': 0, 'prob': 0.9997, 'other': None, 'probs': [0.9997, 0.0001, 0.0002]},
 {'class': 1, 'prob': 0.9998, 'other': None, 'probs': [0.0001, 0.9998, 0.0001]}]
```

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `hyp` | `"The text discusses {}"` | Hypothesis template; `{}` is replaced by the label. |

The positions of the *entailment*, *contradiction* (or *not_entailment*) and *neutral* outputs are
read from the model's `id2label` config, so both 2-class zero-shot models and 3-class NLI models
work. In single-label mode the raw entailment logits of all labels go through a softmax. In
[multi-label mode](#multi-label-mode), each label gets entailment minus contradiction (and
neutral) and then a sigmoid.

### GoalExEntail

Asks an encoder-decoder model a Yes/No question per label and compares the scores of the *Yes* and
*No* tokens at the first decoder step, in one forward pass and without generating text. The prompt
follows the GoalEx property-verification format:

```text
Check whether the TEXT satisfies a PROPERTY. Respond with Yes or No. When uncertain, output No.
Now complete the following example -
input:
PROPERTY: {property}
TEXT: {text}
output:
```

```python
from topicjev.entail import GoalExEntail

model = GoalExEntail("google/flan-t5-base")
with model:
    results = model.entail(docs=docs, lbls=labels)
```

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `template` | the GoalEx prompt above | Prompt with `{property}` and `{text}` placeholders. |
| `reserve` | `8` | Tokens kept free when truncating the document to fit the context. |
| `temperature` | `0.1` | Divides the Yes/No scores before normalization. |

Every spelling of *yes* and *no* that the tokenizer encodes as a single token (`yes`, ` Yes`,
`YES`, ...) is combined with `logsumexp`. A tokenizer without any single-token variant raises a
`ValueError`.

### CausalEntail

The same Yes/No question as `GoalExEntail`, asked to a causal chat model. The prompt is wrapped in
the model's chat template, and the score is read from the logits of the next token.

```python
from topicjev.entail import CausalEntail

model = CausalEntail("Qwen/Qwen2.5-1.5B-Instruct")
with model:
    results = model.entail(docs=docs, lbls=labels)
```

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `template` | the GoalEx prompt | Prompt with `{property}` and `{text}` placeholders. |
| `chat` | `True` | Wrap the prompt in the tokenizer's chat template, if it has one. |
| `thinking` | `False` | Keep "thinking" enabled for reasoning models (e.g. Qwen3); off by default, so the model answers right away. |
| `reserve` | `8` | Tokens kept free when truncating the document. |
| `temperature` | `0.1` | As for `GoalExEntail`. |

The tokenizer is left-padded, so the last position of every row in a batch is the end of the
prompt.

### Seq2SeqEntail

Scores each label by how likely the model is to produce it as the answer to
`"Text: {document}\nThe topic of this text is:"`: the mean log-probability of the label's tokens
under teacher forcing.

```python
from topicjev.entail import Seq2SeqEntail

model = Seq2SeqEntail("google/flan-t5-base")
with model:
    results = model.entail(docs=docs, lbls=labels)
```

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `template` | `"Text: {}\nThe topic of this text is:"` | Prompt with one `{}` placeholder for the document. |
| `reserve` | `50` | Tokens kept free when truncating the document. |

### LayaEntail and JevEntail

[Laya](https://huggingface.co/convaiinnovations/laya-typed-decisions) is an open-source
ModernBERT decision model in the style of TypeSafe's "System One" models: it answers a
multiple-choice question about a text. `LayaEntail` runs it locally; `JevEntail` sends the same
question to the hosted TypeSafe Jev API.

```python
from topicjev.entail import LayaEntail

jev = LayaEntail("convaiinnovations/laya-typed-decisions")
with jev:
    results = jev.entail(docs=docs, lbls=labels)
```

```text
# results, rounded to 4 digits
[{'class': 0, 'prob': 0.8579, 'other': None, 'probs': [0.8579, 0.0350, 0.0340]},
 {'class': 1, 'prob': 0.8544, 'other': None, 'probs': [0.0382, 0.8544, 0.0385]}]
```

**Labels with descriptions.** A label can be a plain name, or a JSON object that maps the name to a
description. Laya uses the description to decide; this is how the
[BERTopic example](../examples/bertopic_laya.md#4-the-bootstrap-re-classify-every-document) passes each topic's
keywords:

```python
import json

labels = [
    json.dumps({"Space Exploration": "rockets, missions, astronauts, the Moon"}),
    json.dumps({"Monetary Policy": "central banks, interest rates, inflation"}),
    json.dumps({"Sports Organization": "leagues, clubs, tournaments"}),
]
```

Use one key per object, and keep label names unique.

**A built-in *Other* option.** Both classes add the choice
`{"Other": "None of the other categories fit this text"}` as a [decoy](#decoys). That is why the
`probs` above sum to about 0.93: the rest went to *Other*. When *Other* wins, `class` is
`len(labels)` and `other` holds that JSON string.

**Always single-label.** The model's choice probabilities are used as they are (with the default
temperature of `1.0`), and `multi_lbl` is ignored.

!!! note "Uncalibrated confidence"
    Confidence calibration has not been tuned for Laya or Jev. The decision is the top category
    plus decoy routing; treat `prob` as an uncalibrated indicator, not as a posterior probability.
    Laya itself warns about this when it loads the `laya-typed-decisions` checkpoint.

| | `LayaEntail` | `JevEntail` |
| :--- | :--- | :--- |
| Runs | Locally, on the [detected device](devices.md) | On the TypeSafe API |
| Constructor | `LayaEntail(model_name, **options)` | `JevEntail(**options)` (no model name) |
| Batching | `batch_size` documents per call | One request per document (large batches not yet evaluated) |
| Credentials | None | `TYPESAFE_API_KEY` in the environment or a `.env` file |

## Scoring and calibration

Every backend goes through the same steps. For documents \(d\), labels \(l\) and decoys:

1. **Pair** every document with every label and decoy, truncating the document to fit the model.
2. **Score** each pair with the backend's raw score \(s(d, l)\).
3. **Calibrate** with [probes](#probes), if given: \(s(d, l) \leftarrow s(d, l) - \frac{1}{|P|}\sum_{p \in P} s(p, l)\).
4. **Scale** by the [temperature](#temperature) \(T\): \(s(d, l) / T\).
5. **Normalize**: a softmax across labels and decoys, or, in [multi-label mode](#multi-label-mode),
   a sigmoid per label.
6. **Decide**: take the top candidate. It becomes the `class` only if it is a real label and its
   probability is above the [threshold](#threshold); otherwise the document is *Other*.

All options below are constructor arguments and work with every backend.

### Decoys

```python
model = CausalEntail("Qwen/Qwen2.5-1.5B-Instruct", decoys=["Celebrity Gossip", "Weather Forecast"])
```

Under a closed list of labels, a softmax always picks one of them, even for a document that fits
none. Decoys are extra, plausible labels that compete for probability. If a decoy wins, the
document is assigned to *Other*, and `other` says which decoy took it. With the three labels above,
the text "Heavy rain and thunderstorms expected over the weekend." comes back as:

```text
{'class': 3, 'prob': 1.0, 'other': 'Weather Forecast', 'probs': [0.0, 0.0, 0.0]}
```

### Probes

```python
model = CausalEntail("Qwen/Qwen2.5-1.5B-Instruct", probes=["", "N/A", "Lorem ipsum dolor sit amet."])
```

Some labels score high for almost any text: the model is biased towards them. Probes are
content-free texts; their mean score per label is subtracted from every document's score before
normalization, which removes that bias (contextual calibration).

### Temperature

`scores / temperature` before the softmax or sigmoid. A lower temperature sharpens the
distribution, a higher one flattens it. `GoalExEntail` and `CausalEntail` default to `0.1`, the
other backends to `1.0`.

!!! tip
    With the default `0.1`, the Yes/No backends often return probabilities of almost exactly 0 or 1
    (in our tests, `CausalEntail` with Qwen2.5-1.5B gave `prob=1.0` on clear-cut examples). That
    is fine for picking a label. If you want to rank documents by confidence or compare against a
    threshold, raise the temperature, e.g. `temperature=1.0`.

### Threshold

The top label is accepted only if its probability is **above** the threshold. The default depends
on the mode:

- single-label: \(2 / N\) for \(N\) labels, i.e. twice the uniform probability;
- multi-label: \(0.5\).

!!! warning "With one or two labels, the default threshold rejects everything"
    For \(N = 2\) the default is \(2/2 = 1.0\), and no probability is ever above it, so **every
    document becomes *Other***, even at 99.99% confidence. Pass an explicit value, e.g.
    `threshold=0.5`. To check a **single** label, use `multi_lbl=True` instead: a softmax over one
    label is always `1.0`, while the sigmoid of multi-label mode judges the label on its own.

### Multi-label mode

```python
model = XEncoderEntail("MoritzLaurer/deberta-v3-base-zeroshot-v2.0", multi_lbl=True)
```

With `multi_lbl=True`, every label is scored independently with a sigmoid, so several labels can
have high probability at once. `class` still reports only the best label; read `probs` to get all
labels that apply:

```python
assigned = [[labels[i] for i, p in enumerate(r["probs"]) if p > 0.5] for r in results]
```

`LayaEntail` and `JevEntail` are always single-label.

## Common parameters

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `batch_size` | `16` | Pairs per forward pass (documents per call for Laya). |
| `max_tokens` | `512` | Context length to use when neither the tokenizer nor the model config states one. |
| `multi_lbl` | `False` | Sigmoid per label instead of a softmax across labels. |
| `decoys` | `None` | Extra labels that send a document to *Other* when they win. |
| `probes` | `None` | Content-free texts used to subtract label bias. |
| `temperature` | `1.0` (`0.1` for GoalEx and Causal) | Divides the scores before normalization. |
| `threshold` | `2 / N` (single-label) or `0.5` (multi-label) | Minimum probability to accept the top label. |

## Long documents

Documents are truncated so that the prompt, the longest label and the answer fit the model's
context length. That length comes from the tokenizer (`model_max_length`) or the model config, and
falls back to `max_tokens`. Most cross-encoders accept 512 tokens, so a long document is judged by
its beginning only. [Compress](compression.md) long documents first if their main point may come
later.

## Writing your own backend

Subclass `Entailment` (or `LocalEntailment` for a PyTorch model, which adds device detection and
memory clean-up) and implement three methods:

- `load_model()`: load weights or open a client;
- `prep_pairs(prompts, lbls)`: build one `Pair` per (document, label), documents in the outer loop;
- `batch_score(pairs)`: return one raw score per pair (higher means more likely).

Probes, decoys, temperature, normalization and thresholds then work unchanged. A toy backend that
counts shared words:

```python
from topicjev.entail import Entailment, Pair


class KeywordEntail(Entailment):
    """Scores a label by how many of its words appear in the document."""

    def load_model(self) -> None:
        pass  # nothing to load

    def prep_pairs(self, prompts: list[str], lbls: list[str]) -> list[Pair]:
        return [Pair(inpt=doc.lower(), targ=lbl.lower()) for doc in prompts for lbl in lbls]

    def batch_score(self, pairs: list[Pair]) -> list[float]:
        return [float(sum(w in p.inpt for w in p.targ.split())) for p in pairs]


model = KeywordEntail(None, temperature=0.5)
with model:
    results = model.entail(
        docs=["Interest rates and inflation worry central banks."],
        lbls=["space rockets", "interest rates inflation", "football league"],
    )
print(results)
# [{'class': 1, 'prob': 0.995..., 'other': None, 'probs': [0.0025, 0.995..., 0.0025]}]
```
