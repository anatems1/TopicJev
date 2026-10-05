# Compression

`topicjev.compress` shortens documents before they are classified, embedded or labeled. Small
classifiers read only the first few hundred tokens of a text (see
[Long documents](entailment.md#long-documents)), and LLM prompts that summarize a whole cluster get
expensive quickly. A compressed document keeps its main point within that budget.

All compressors take a list of strings and return a list of strings of the same length:

```python
with compressor:
    short_docs = compressor.compress(docs)
```

## Choosing a compressor

| Compressor | Mechanism | Output length | Good for |
| :--- | :--- | :--- | :--- |
| [`CausalCompressor`](#causalcompressor) | Summary by a causal chat LM | Up to `ratio` × prompt length, in tokens | Modern instruction-tuned LLMs (Qwen, Llama, Mistral); the budget grows with the document. |
| [`CNNCompressor`](#cnncompressor) | Summary by a summarization-tuned seq2seq model | Up to `ratio` × the model's context length | Checkpoints fine-tuned on news summarization, such as BART-CNN or PEGASUS. |
| [`Seq2SeqCompressor`](#seq2seqcompressor) | Instruction-prompted seq2seq summary | Up to `ratio` × the model's context length | General instruction-tuned encoder-decoders such as FLAN-T5. |
| [`LinguaCompressor`](#linguacompressor) | Token pruning with LLMLingua-2, no generation | About `ratio` of the original tokens | Fast, extractive compression that keeps the original wording. |

Parameters shared by all compressors:

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `ratio` | `0.3` | Output budget; its meaning depends on the compressor, see the table above. |
| `batch_size` | `8` | Documents per generation batch (not used by `LinguaCompressor`). |
| `source_len` | per compressor | Input length to use when the tokenizer and model config do not state one. |

The examples below use these two documents:

```python
long_docs = [
    "Researchers developed a solid-state lithium battery with a ceramic electrolyte that resists dendrite growth. "
    "Over 1,000 charge cycles the cell retained 92 percent of its capacity, and it operated safely at temperatures "
    "up to 80 degrees Celsius. The authors argue the design could shorten charging times for electric vehicles, "
    "although manufacturing the thin ceramic layers at scale remains expensive.",
    "A city council approved a plan to electrify its entire bus fleet by 2030. The plan replaces 400 diesel buses, "
    "adds charging depots at three garages, and is funded partly by a federal grant. Officials expect lower "
    "maintenance costs and cleaner air along busy corridors, while unions asked for retraining programs for mechanics.",
]
```

## CausalCompressor

Prompts a causal chat model to summarize each document. The token budget is relative to the input:
`max_new_tokens = ratio × prompt length`, where the prompt length is the longest prompt in the
batch, including the instruction and chat template.

```python
from topicjev.compress import CausalCompressor

compressor = CausalCompressor("Qwen/Qwen2.5-1.5B-Instruct", ratio=0.3, batch_size=4)
with compressor:
    summaries = compressor.compress(long_docs)
```

```text
Researchers created a solid-state lithium battery with a ceramic electrolyte that prevents
dendrite growth and maintains over 92% capacity after 1,000 charge cycles. It operates safely up to
```

!!! warning "The budget is a hard cut"
    Generation stops when the budget runs out, even mid-sentence, as in the output above. Raise
    `ratio`, or set `min_len`, if summaries come back truncated.

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `instruct` | see below | Prompt template with a `{text}` placeholder. |
| `chat` | `True` | Wrap the prompt in the tokenizer's chat template. |
| `thinking` | `False` | Keep "thinking" enabled for reasoning models. |
| `min_len` | `0` | Minimum number of new tokens; also a floor for the budget. |
| `num_beams` | `1` | Beam search width (`1` = greedy decoding). |
| `source_len` | `2048` | Fallback input length. |

The default instruction:

```text
Summarize the text below in several declarative sentences in paragraph form. Output only the summary, with no preamble.

TEXT:
{text}

SUMMARY:
```

## CNNCompressor

For seq2seq checkpoints fine-tuned to summarize, such as
[`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn). The document goes in
as is, without an instruction, and the summary is generated with beam search. The budget is fixed
per model: `max_length = ratio × context length` (1024 tokens for BART).

```python
from topicjev.compress import CNNCompressor

compressor = CNNCompressor("facebook/bart-large-cnn", ratio=0.1, min_len=20)
with compressor:
    summaries = compressor.compress(long_docs)
```

```text
Researchers developed a solid-state lithium battery with a ceramic electrolyte that resists
dendrite growth. Over 1,000 charge cycles the cell retained 92 percent of its capacity. Design
could shorten charging times for electric vehicles.
```

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `min_len` | `30` | Minimum summary length, in tokens. |
| `num_beams` | `4` | Beam search width. |
| `source_len` | `512` | Fallback input length. |

## Seq2SeqCompressor

The same as `CNNCompressor`, with an instruction in front of each document, for general
instruction-tuned models such as FLAN-T5.

```python
from topicjev.compress import Seq2SeqCompressor

compressor = Seq2SeqCompressor("google/flan-t5-base", ratio=0.2)
with compressor:
    summaries = compressor.compress(long_docs)
```

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `instruct` | `"Summarize the following text in two or three plain sentences. {}"` | Instruction; `{}` is replaced by the document. |
| `min_len` | `30` | Minimum summary length, in tokens. |
| `num_beams` | `4` | Beam search width. |
| `source_len` | `1024` | Fallback input length. |

!!! note
    Small abstractive models can add details that are not in the text. In our test, `flan-t5-base`
    summarized the bus-fleet document as being about New York City, which the document never
    mentions. Check a sample of summaries before building on them, or use `LinguaCompressor`,
    which only deletes tokens.

## LinguaCompressor

[LLMLingua-2](https://github.com/microsoft/LLMLingua) classifies every token as *keep* or *drop*
and deletes the dropped ones. Nothing is generated, so the result keeps the original wording and
cannot invent content. It needs the `lingua` extra:

```bash
pip install -e ".[lingua]"
```

```python
from topicjev.compress import LinguaCompressor

compressor = LinguaCompressor(
    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    ratio=0.5,
)
with compressor:
    pruned = compressor.compress(long_docs)
```

```text
developed solid - state lithium battery ceramic electrolyte resists dendrite growth.   Over 1, 000
charge cycles cell retained 92 percent capacity,   operated safely temperatures 80 degrees Celsius.
authors design shorten charging times electric vehicles,   manufacturing thin ceramic layers scale expensive.
```

The output is not fluent prose, but it keeps the original words and every number. Here `ratio` is the share of tokens to keep. For better quality at a higher cost, use
`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`.

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `target_token` | `-1` | Target length in tokens; `-1` uses `ratio` instead. |
| `force_tokens` | newlines, space, `. , : ; ! ?` | Tokens that are always kept. |
| `force_digits` | `False` | Always keep tokens that contain digits. |
| `drop_consecutive` | `True` | Drop repeated force tokens that end up next to each other. |

LLMLingua runs on a CUDA GPU when one is available and on the CPU otherwise, including on Apple
Silicon Macs.

## Compress, then classify

```python
from topicjev.compress import CNNCompressor
from topicjev.entail import XEncoderEntail

with CNNCompressor("facebook/bart-large-cnn", ratio=0.1) as compressor:
    short_docs = compressor.compress(long_docs)

with XEncoderEntail("MoritzLaurer/deberta-v3-base-zeroshot-v2.0") as model:
    results = model.entail(docs=short_docs, lbls=["Battery Technology", "Public Transit", "Sports"])
```

Each `with` block frees its model before the next one is loaded, so the two never share GPU memory.
