# LLM Generation

`LocalGenerator` runs a local causal language model (Qwen, Llama, Mistral, ...) over a batch of
inputs. Its main use in TopicJev is structured output: asking the model for a JSON object, such as a
topic's title and keywords, and retrying until the answer parses.

## Run a template over a batch

A prompt is a Python format string. Each input is a dictionary whose keys fill the placeholders:

```python
from topicjev.gen import LocalGenerator

gen = LocalGenerator("Qwen/Qwen2.5-1.5B-Instruct", json_mode=True, temperature=0.0)

prompt = (
    "Extract key metadata from the document below as JSON with keys 'title' and 'keywords':\n\n"
    "TEXT:\n{text}\n\nJSON:"
)
docs = [
    {"text": "CRISPR gene editing shows promising results in clinical trials for sickle cell disease."},
    {"text": "Quantum computing algorithms achieve polynomial speedups for specific simulation tasks."},
]

with gen:
    results = gen.batch(prompt, docs, req_keys=["title", "keywords"])

for item in results:
    print(item["title"], item["keywords"])
```

```text
CRISPR Gene Editing Shows Promising Results in Clinical Trials for Sickle Cell Disease ['CRISPR', 'gene editing', 'clinical trials', 'sickle cell disease']
Quantum Computing Algorithms Achieve Polynomial Speedups ['quantum computing', 'algorithms', 'polynomial speedups', 'simulation tasks']
```

Each prompt is wrapped in the model's chat template before generation. To put literal braces in a
template, for example a JSON example, double them: `{{"title": "..."}}`.

## JSON mode and retries

With `json_mode=True` (the default), every answer goes through [`clean_json`](#clean_json) and comes
back as a `dict`. With `req_keys`, the answers are also validated:

1. All inputs are generated in batches.
2. An answer that does not parse, or lacks one of `req_keys`, is generated again.
3. After `max_retries` rounds (default `5`), inputs that still fail get an empty dict `{}`.

Without `req_keys` there are no retries: an answer that does not parse becomes `{}`. Either way,
the result list has one entry per input, in order, and you only need to check for `{}`:

```python
failed = [i for i, r in enumerate(results) if not r]
```

Retries only help when sampling is on (`temperature > 0`). With `temperature=0.0`, generation is
greedy and a retry usually produces the same answer again.

With `json_mode=False`, `batch()` returns the generated strings unchanged (stripped of surrounding
whitespace), and `req_keys` is ignored.

## A single prompt

```python
with gen:
    print(gen.query("Return a JSON object with a single key 'answer' holding the capital of France."))
# {'answer': 'Paris'}
```

`query(user_prompt, sys_prompt="You are a helpful AI assistant")` sends one prompt with a system
prompt. `run_chain()` is an alias of `batch()`.

## Parameters

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `temperature` | `0.1` | `0` for greedy decoding; above `0`, sampling with `top_k=20` and `top_p=0.95`. |
| `max_tokens` | `4000` | Maximum number of new tokens per answer. |
| `max_input_tokens` | `4000` | Prompts are truncated to this many tokens. |
| `json_mode` | `True` | Parse answers as JSON dictionaries. |
| `batch_size` | `4` | Prompts per generation batch. |
| `max_retries` | `5` | Retry rounds for answers that fail validation (with `req_keys`). |

On a CUDA GPU, the model is compiled with `torch.compile` when it is loaded, so the first batch
takes longer while it compiles.

!!! note "Reasoning models"
    Chat templates that support it (e.g. Qwen3) are applied with thinking turned off, so the model
    answers with the JSON object right away instead of a long reasoning trace first. The
    [KMeans + LLM labels example](../examples/kmeans_genlbs_laya.md) relies on this with
    `Qwen/Qwen3-1.7B`.

## clean_json

The parser behind JSON mode is also available on its own. It accepts the shapes LLMs typically
produce: bare JSON, JSON inside a Markdown code fence, or JSON surrounded by other text. It returns
`{}` if nothing parses, or if the JSON is not an object.

```python
from topicjev.gen import clean_json

clean_json('Sure! Here it is:\n```json\n{"title": "Bees"}\n```')   # {'title': 'Bees'}
clean_json('The answer is {"title": "Bees"} as requested.')         # {'title': 'Bees'}
clean_json("no json here")                                          # {}
```

## Next

The [bundled prompt templates](prompts.md) use `LocalGenerator` to turn a cluster of documents into
a title, a summary and keywords.
