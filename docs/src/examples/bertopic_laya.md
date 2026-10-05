# BERTopic + Laya

This walkthrough runs
[`examples/bertopic_laya.py`](https://github.com/anatems1/TopicJev/blob/master/examples/bertopic_laya.py).
It fits a BERTopic model on the first 1,500 articles of
[AG News](https://huggingface.co/datasets/fancyzhx/ag_news), then asks the Laya decision model to
re-assign every article to one of the topics, using only each topic's name and keywords. Where Laya
agrees with BERTopic, the topic is well described; where it disagrees, the topic needs work.

## Run it

The example needs the `examples` extra (BERTopic, pandas, scikit-learn, datasets). From the
repository root:

```bash
pip install -e ".[all]"
python examples/bertopic_laya.py
```

The first run downloads the dataset slice and two models, embeds the articles, fits the topic model
and classifies every article; with the models already downloaded, this took under two minutes on
an Apple Silicon Mac. Results are
written to `output/<hash>/` in the current directory. Later runs with the same settings reuse the
saved results and finish in seconds.

## Step by step

### 1. Settings and the output folder

All settings are constants at the top of the script. Results go to `output/<hash>/`, where the
hash is computed from the settings, so changing any of them starts a fresh run instead of reusing
stale files:

```python
DATASET_NAME: str = "fancyzhx/ag_news"
N_SAMPLES: int = 1500
EMBED_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
EMBED_PREFIX: str = "clustering: "
JEV_MODEL: str = "convaiinnovations/laya-typed-decisions"
N_TOPICS: int = 7
RND_SEED: int = 42

# output dir is keyed by CONFIG: add any setting you vary here, or cached results get reused
CONFIG = {
    "dataset": DATASET_NAME,
    "n_samples": N_SAMPLES,
    "embed_model": EMBED_MODEL,
    "embed_prefix": EMBED_PREFIX,
    "jev_model": JEV_MODEL,
    "n_topics": N_TOPICS,
    "seed": RND_SEED,
}
OUT_DIR: Path = Path("./output") / hashlib.sha1(json.dumps(CONFIG).encode()).hexdigest()[:8]
```

!!! tip
    If you change anything else in the script, such as the BERTopic settings, either add it to
    `CONFIG` or delete the output folder. Otherwise the cached results of the old run are reused.

### 2. The data

The first `N_SAMPLES` articles of the AG News training split are loaded with Hugging Face
`datasets` and saved to `docs.csv`, so later runs do not need the network:

```python
ds = load_dataset(DATASET_NAME, split=f"train[:{N_SAMPLES}]")
docs_df = ds.to_pandas()
docs_df["id"] = range(len(docs_df))
docs_df.to_csv(OUT_DIR / "docs.csv", index=False)
```

AG News also has a gold `label` column (World, Sports, Business, Sci/Tech). The pipeline does not
use it, but it is kept in the CSV files, which makes it easy to sanity-check the results below.

### 3. The topic model

A standard BERTopic pipeline, made reproducible: UMAP is seeded, and KMeans replaces HDBSCAN so
that the number of topics is fixed and no document is left as an outlier.

```python
tmodel = BERTopic(
    nr_topics=N_TOPICS,
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=RND_SEED),
    hdbscan_model=KMeans(n_clusters=N_TOPICS, random_state=RND_SEED),
    vectorizer_model=CountVectorizer(max_df=0.8, stop_words="english", ngram_range=(1, 2)),
    ctfidf_model=ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True),
    representation_model=MaximalMarginalRelevance(diversity=0.3),
)
```

[`LocalEmbedder`](../getting_started/embeddings.md) embeds the articles with
[nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), using the
`clustering: ` task prefix that model expects. With `save_dir`, the vectors are written to
`embeds.bin` in the output folder and loaded from there on the next run.

```python
embedder = LocalEmbedder(
    EMBED_MODEL,
    prefix=EMBED_PREFIX,
    trust_remote_code=False,  # use transformers' built-in NomicBert
    save_dir=str(OUT_DIR),
)
embeds = embedder.get_embeds(docs)
tmodel.fit_transform(docs, embeddings=embeds)
```

`trust_remote_code=False` matters: the checkpoint's own modeling code predates transformers 5 and
fails with it (see the [FAQ](../faq.md#nomicbertmodel-object-has-no-attribute-get_extended_attention_mask)).

### 4. The bootstrap: re-classify every document

Each topic becomes one candidate label for
[`LayaEntail`](../getting_started/entailment.md#layaentail-and-jeventail): a JSON object that maps
the topic's name to its top five keywords. Laya reads the keywords as the description of the class.

```python
jev = LayaEntail(JEV_MODEL)
with jev:
    results = jev.entail(
        docs=docs,
        lbls=[
            json.dumps({row["Name"]: ", ".join(ast.literal_eval(row["Representation"])[:5])})
            for _, row in tinfo.iterrows()
        ],
    )
docs_df["jev"] = [r["class"] for r in results]
```

For topic 0, the label is:

```json
{"0_athens_olympic_team_phelps": "athens, olympic, team, phelps, gold"}
```

`results[i]["class"]` is an index into the list of labels, which follows the rows of
`get_topic_info()`. Because KMeans produces topics `0..N-1` without the outlier topic `-1`, row
`i` is topic `i`, and the class index can be compared with the topic id directly. An article that
fits none of the topics gets class `N_TOPICS` (here `7`), meaning *Other*.

!!! note "Using HDBSCAN instead of KMeans"
    HDBSCAN puts outliers in topic `-1`, which is then the first row of `get_topic_info()`. Either
    leave that row out of the labels, or map class indices back to topic ids through
    `tinfo["Topic"]` before comparing.

### 5. Agreement per topic

```python
for topic_id in docs_df["topic_id"].unique():
    topic_docs = docs_df[docs_df["topic_id"] == topic_id]
    agreement = (topic_docs["topic_id"] == topic_docs["jev"]).mean()
    print(f"Topic {topic_id}: Agreement between BERTopic and Laya: {agreement:.2%}")
```

## Reading the results

One run on an Apple Silicon Mac printed the following; exact numbers vary with hardware and library
versions.

```text
Topic 4: Agreement between BERTopic and Laya: 60.78%
Topic 2: Agreement between BERTopic and Laya: 73.32%
Topic 1: Agreement between BERTopic and Laya: 77.98%
Topic 3: Agreement between BERTopic and Laya: 18.40%
Topic 5: Agreement between BERTopic and Laya: 72.04%
Topic 0: Agreement between BERTopic and Laya: 35.77%
Topic 6: Agreement between BERTopic and Laya: 86.11%
```

Agreement alone says *how much* a topic holds together. A cross-tabulation of the saved labels
also shows *where* the rejected documents went (rows: BERTopic topics with their first keywords;
columns: Laya's classes):

```python
import pandas as pd

df = pd.read_csv("output/<hash>/jev_labels.csv")
print(pd.crosstab(df["topic_id"], df["jev"], margins=True))
```

<div class="tj-compact" markdown>

| Topic ↓ · class → | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 *Other* | All |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 athens, olympic | **98** | 11 | 3 | 0 | 0 | 2 | 0 | 160 | 274 |
| 1 najaf, iraqi | 3 | **131** | 3 | 1 | 1 | 5 | 0 | 24 | 168 |
| 2 google, software | 0 | 6 | **305** | 2 | 7 | 15 | 0 | 81 | 416 |
| 3 hurricane, bush | 0 | 41 | 3 | **30** | 1 | 4 | 0 | 84 | 163 |
| 4 prices, oil prices | 0 | 15 | 14 | 3 | **141** | 0 | 3 | 56 | 232 |
| 5 space, scientists | 1 | 5 | 9 | 3 | 0 | **152** | 0 | 41 | 211 |
| 6 chavez, venezuela | 0 | 2 | 0 | 0 | 0 | 0 | **31** | 3 | 36 |
| All | 102 | 211 | 337 | 39 | 150 | 178 | 34 | 449 | 1500 |

</div>

What this run suggests:

- **Topic 6** (Chavez, Venezuela, referendum) is small and coherent: 86% of its articles are
  confirmed.
- **Topic 0** (athens, olympic, team, phelps, gold) keeps only 98 of 274 articles and sends 160 to
  *Other*. The gold labels explain why: 88% of the articles sent to *Other* are Sports, but only
  10% of them mention the Olympics or Athens, against 70% of the articles Laya kept. The keywords
  describe the Olympic part of a broader sports topic; a description such as "sports" would fit
  the cluster better.
- **Topic 3** (hurricane, bush, charley) is the weakest, at 18%. Only a fifth of its articles
  mention a hurricane or storm, and the 30 that Laya keeps all do. The rest is a mix of world news
  (US politics, East Asian diplomacy and more; 113 of the 163 articles carry the gold label World)
  that goes to the Iraq topic or to *Other*. The keywords describe only the hurricane part, so the
  cluster is a candidate to split.
- **Other** (class 7) collects 449 articles, 30% of the slice. Five keywords per topic are a thin
  description; LLM-written titles and descriptions, as in the
  [KMeans + LLM labels example](kmeans_genlbs_laya.md), may give the classifier more to work with.

To check findings like these against the gold labels yourself:

```python
print(pd.crosstab(df["topic_id"], df["label"]))  # 0 World, 1 Sports, 2 Business, 3 Sci/Tech
```

## Next steps

- Describe topics better: generate a title and a one-sentence description per topic with
  [`LocalGenerator`](../getting_started/generation.md) and the bundled
  [prompts](../getting_started/prompts.md), then pass those to `LayaEntail` instead of keywords.
  The [KMeans + LLM labels example](kmeans_genlbs_laya.md) does exactly that.
- Try another classifier: [`XEncoderEntail`](../getting_started/entailment.md#xencoderentail)
  checks claims such as "The text discusses the Athens Olympics".
- Shorten long documents first with a [compressor](../getting_started/compression.md).

## Full script

??? example "examples/bertopic_laya.py"

    ```python
    --8<-- "examples/bertopic_laya.py"
    ```
