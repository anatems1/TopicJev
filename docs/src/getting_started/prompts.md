# Prompt Templates

`topicjev.prompts` ships prompt templates for describing topics with an LLM. They are plain text
files inside the package, loaded by name:

```python
from topicjev.prompts import list_prompts, load_prompt

list_prompts()             # ['classes', 'reduce']
template = load_prompt("reduce")
```

Both templates are format strings for [`LocalGenerator.batch()`](generation.md), and both ask for a
bare JSON object.

## `reduce`: from facts to a theme

Input: short factual snippets from the documents of one cluster, for example the
[compressed](compression.md) documents, or the documents nearest the cluster's centroid. Output: the *one* dominant theme of the snippets, as a
title, a literature-review style summary, and exactly five keywords. The prompt tells the model to
ignore snippets that do not fit the main theme instead of stitching every snippet into the summary.

| Placeholder | Fill with |
| :--- | :--- |
| `{text}` | A JSON object `{"facts": ["...", "..."]}` |

Output keys: `title`, `summary`, `keywords`.

## `classes`: from a theme to a checkable class name

Input: the output of `reduce`, plus the titles already used by other topics. Output: a 2–5 word
class name and a one-sentence description. The title is designed to complete the sentence
*"The text discusses ______"*, which is exactly the default hypothesis of
[`XEncoderEntail`](entailment.md#xencoderentail). The prompt therefore asks for a single, concrete,
checkable claim: never two topics joined by "and", never a generic umbrella term, and distinct from
every used title.

| Placeholder | Fill with |
| :--- | :--- |
| `{text}` | The `reduce` result, as JSON |
| `{used_titles}` | Titles already given to other topics, one per line (can be empty) |

Output keys: `title`, `desc`.

## Example: name a cluster, then check it

The [KMeans + LLM labels example](../examples/kmeans_genlbs_laya.md) runs this chain on real
clusters: it feeds the five documents nearest each centroid to `reduce`, passes the result to
`classes`, and hands the titles and descriptions to `LayaEntail`. The steps in isolation:

```python
import json

from topicjev.gen import LocalGenerator
from topicjev.prompts import load_prompt

facts = {
    "facts": [
        "Solid-state lithium cells with ceramic electrolytes resist dendrite growth.",
        "A sulfide electrolyte raised ionic conductivity at room temperature.",
        "Ceramic separators kept 92% capacity after 1,000 cycles.",
        "A survey reports rising cobalt prices.",
    ]
}

gen = LocalGenerator("Qwen/Qwen2.5-1.5B-Instruct", temperature=0.0)
with gen:
    theme = gen.batch(
        load_prompt("reduce"),
        [{"text": json.dumps(facts)}],
        req_keys=["title", "summary", "keywords"],
    )[0]
    label = gen.batch(
        load_prompt("classes"),
        [{"text": json.dumps(theme), "used_titles": "Bus Fleet Electrification"}],
        req_keys=["title", "desc"],
    )[0]

print(theme["title"])   # Electrolyte and Separator Improvements
print(label["title"])   # Lithium Cell Dendrite Growth Prevention
```

The class title can now be checked against every document of the cluster. With a single label,
use `multi_lbl=True`: the label is then judged on its own, with a sigmoid and a threshold of `0.5`.
(In single-label mode, a softmax over one label always gives `1.0`.)

```python
from topicjev.entail import XEncoderEntail

cluster_docs = [
    "Solid-state lithium cells with ceramic electrolytes resist dendrite growth.",
    "Ceramic separators kept 92% capacity after 1,000 cycles.",
    "A survey reports rising cobalt prices.",
]
with XEncoderEntail("MoritzLaurer/deberta-v3-base-zeroshot-v2.0", multi_lbl=True) as model:
    results = model.entail(docs=cluster_docs, lbls=[label["title"]])

for doc, res in zip(cluster_docs, results):
    print(res["class"], round(res["probs"][0], 4), doc)
```

```text
0 0.9991 Solid-state lithium cells with ceramic electrolytes resist dendrite growth.
1 0.0002 Ceramic separators kept 92% capacity after 1,000 cycles.
1 0.0003 A survey reports rising cobalt prices.
```

Class `0` confirms the document; class `1` (= `len(lbls)`) is *Other*. Only the first document is
confirmed: the title is precise enough to reject the cobalt-price outlier, and also narrow enough
to miss the capacity result. That trade-off is what the bootstrap measures.

!!! note "Check what small models return"
    In our tests a 1.5B model always returned valid JSON, but did not always follow the finer rules: the `reduce`
    title above, "Electrolyte and Separator Improvements", joins two topics with "and", which the
    prompt forbids. Review a sample of the outputs, or try a larger model.

## Your own templates

Any string works as a template for `LocalGenerator.batch()`. Keep two rules in mind:

- Placeholders are filled with `str.format(**input)`, so every `{name}` must be a key of each input
  dictionary.
- Literal braces must be doubled: write `{{"title": "..."}}` to show the model a JSON example.
