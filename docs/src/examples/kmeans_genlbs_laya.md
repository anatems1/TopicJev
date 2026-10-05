# KMeans + LLM Labels + Laya

This walkthrough runs
[`examples/kmeans_genlbs_laya.py`](https://github.com/anatems1/TopicJev/blob/master/examples/kmeans_genlbs_laya.py),
a lightweight pipeline that does not need BERTopic. It clusters the same 1,500 AG News articles as
the [BERTopic example](bertopic_laya.md) with plain KMeans, asks a local LLM (`Qwen3-1.7B`) to
write a checkable label for every cluster, and then lets Laya verify each article against those
labels.

Where the BERTopic example describes topics by five keywords, this one describes them with an
LLM-written title and a one-sentence description, built with the bundled
[`reduce` and `classes` prompts](../getting_started/prompts.md).

## Run it

From the repository root:

```bash
pip install -e ".[all]"
python examples/kmeans_genlbs_laya.py
```

The first run downloads the dataset slice and three models (the embedder, `Qwen/Qwen3-1.7B` and
Laya). Results are written to `output/<hash>/`; later runs with the same settings reuse them.

!!! bug "Segmentation fault on macOS"
    On macOS, the script can crash with `Segmentation fault: 11` (or
    `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`) at
    `embedder.index.search(...)` in step 4. FAISS and PyTorch each bundle their own copy of the
    OpenMP runtime, and FAISS's multi-threaded search crashes when both are loaded. Run with a
    single OpenMP thread:

    ```bash
    OMP_NUM_THREADS=1 python examples/kmeans_genlbs_laya.py
    ```

    The LLM and Laya still run on the GPU (MPS); only CPU-side OpenMP work is single-threaded. The
    [BERTopic example](bertopic_laya.md) is not affected, because it never searches the index.

## Step by step

### 1–2. Data and embeddings

The same as in the BERTopic example: the first 1,500 AG News training articles, embedded with
`nomic-embed-text-v1.5` and cached in a FAISS index (`embeds.bin`) by
[`LocalEmbedder`](../getting_started/embeddings.md).

### 3. Cluster with KMeans

```python
kmeans = KMeans(n_clusters=N_TOPICS, random_state=RND_SEED)
cluster_labels = kmeans.fit_predict(embeds)
docs_df["topic_id"] = cluster_labels
```

### 4. Label every cluster with an LLM

For each cluster, the script averages its embeddings into a centroid, L2-normalizes it, and asks
the embedder's FAISS index for the five nearest articles. Because the vectors are normalized, the
index's inner product is the cosine similarity:

```python
centroids = np.array([embeds[docs_df["topic_id"] == k].mean(axis=0) for k in range(N_TOPICS)])
norms = np.linalg.norm(centroids, axis=1, keepdims=True)
norms[norms == 0] = 1.0
normalized_centroids = (centroids / norms).astype(np.float32)
_, top_5_indices = embedder.index.search(normalized_centroids, 5)

embedder.close()  # free the embedder before loading the LLM
```

The five articles go to the [`reduce` prompt](../getting_started/prompts.md#reduce-from-facts-to-a-theme),
which names the cluster's dominant theme with a title, a summary and five keywords. That result
goes to the [`classes` prompt](../getting_started/prompts.md#classes-from-a-theme-to-a-checkable-class-name),
which turns it into a short, checkable class title and a one-sentence description. The titles of
earlier clusters are passed as `used_titles`, so the LLM can avoid reusing them (simplified from
the script, which also falls back to the `reduce` title if a call fails):

```python
gen = LocalGenerator(LLM_MODEL, json_mode=True, temperature=0.1, max_tokens=512)

with gen:
    for topic_id in range(N_TOPICS):
        top_docs = [docs[idx] for idx in top_5_indices[topic_id]]
        reduce_out = gen.batch(
            reduce_tpl,
            [{"text": json.dumps({"facts": top_docs}, indent=2)}],
            req_keys=["title", "summary", "keywords"],
        )[0]
        class_out = gen.batch(
            classes_tpl,
            [{"text": json.dumps(reduce_out, indent=2), "used_titles": used_titles_str}],
            req_keys=["title", "desc"],
        )[0]
        used_titles.append(class_out["title"])
```

`LocalGenerator` retries an answer until it parses as JSON with the required keys (see
[JSON mode and retries](../getting_started/generation.md#json-mode-and-retries)), and turns
Qwen3's thinking mode off, so the 512-token budget goes to the answer.

### 5. Verify with Laya

Each topic becomes a Laya criterion that maps the class title to its description and keywords:

```python
criteria_text = f"{row['Description']} (Keywords: {', '.join(kw_list)})"
topic_criteria.append(json.dumps({row["Name"]: criteria_text}))

results = jev.entail(docs=docs, lbls=topic_criteria)
```

### 6. Agreement per topic

As in the BERTopic example: the share of each cluster's articles that Laya assigns back to it.

## Reading the results

One run on an Apple Silicon Mac (2 min 39 s with the models already downloaded) produced these
labels and agreement rates; exact numbers and generated labels vary with hardware and library
versions.

| Cluster | Generated class title | Articles | Agreement |
| :--- | :--- | ---: | ---: |
| 0 | Baseball Game Outcomes and Performances | 182 | 82.42% |
| 1 | Oil Price Drop-Induced Stock Market Recovery | 163 | 87.73% |
| 2 | Ocean Dead Zone Monitoring | 280 | 26.07% |
| 3 | Peace Bid in Najaf | 277 | 33.94% |
| 4 | Regulatory Approval for Public Offering | 195 | 26.67% |
| 5 | Windows Security Patching Tools | 290 | 46.55% |
| 6 | U.S. Swimming Performance in Olympic Competition | 113 | 22.12% |

??? example "Console output (descriptions shortened)"

    ```text
    --- Discovered Topics ---
    Topic 0 (n=182): Baseball Game Outcomes and Performances - Baseball game outcomes and performances are examined, ...
    Topic 1 (n=163): Oil Price Drop-Induced Stock Market Recovery - Oil price drops drive U.S. stock market recovery ...
    Topic 2 (n=280): Ocean Dead Zone Monitoring - Ocean dead zones are studied through advanced environmental monitoring ...
    Topic 3 (n=277): Peace Bid in Najaf - The Iraqi Peace Mission in Najaf aims to end a radical Shi'ite uprising ...
    Topic 4 (n=195): Regulatory Approval for Public Offering - Regulatory approval is sought in the final stages of Google's IPO ...
    Topic 5 (n=290): Windows Security Patching Tools - Windows security patching tools are used to address vulnerabilities ...
    Topic 6 (n=113): U.S. Swimming Performance in Olympic Competition - The U.S. swimming team faced challenges in key events ...

    --- Topic Alignment & Agreement ---
    Topic 0 [Baseball Game Outcomes and Performances]: Agreement between KMeans and Laya: 82.42%
    Topic 1 [Oil Price Drop-Induced Stock Market Recovery]: Agreement between KMeans and Laya: 87.73%
    Topic 2 [Ocean Dead Zone Monitoring]: Agreement between KMeans and Laya: 26.07%
    Topic 3 [Peace Bid in Najaf]: Agreement between KMeans and Laya: 33.94%
    Topic 4 [Regulatory Approval for Public Offering]: Agreement between KMeans and Laya: 26.67%
    Topic 5 [Windows Security Patching Tools]: Agreement between KMeans and Laya: 46.55%
    Topic 6 [U.S. Swimming Performance in Olympic Competition]: Agreement between KMeans and Laya: 22.12%
    ```

Where the articles went (rows: KMeans clusters; columns: Laya's classes):

<div class="tj-compact" markdown>

| Cluster ↓ · class → | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 *Other* | All |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 Baseball | **150** | 0 | 0 | 1 | 0 | 0 | 8 | 23 | 182 |
| 1 Oil prices, stocks | 1 | **143** | 1 | 2 | 0 | 0 | 0 | 16 | 163 |
| 2 Ocean dead zones | 10 | 3 | **73** | 5 | 0 | 5 | 0 | 184 | 280 |
| 3 Najaf | 18 | 9 | 2 | **94** | 0 | 3 | 0 | 151 | 277 |
| 4 Public offering | 13 | 18 | 0 | 1 | **52** | 7 | 0 | 104 | 195 |
| 5 Windows security | 7 | 2 | 0 | 2 | 10 | **135** | 0 | 134 | 290 |
| 6 Olympic swimming | 57 | 0 | 0 | 5 | 0 | 1 | **25** | 25 | 113 |
| All | 256 | 175 | 76 | 110 | 62 | 151 | 33 | 637 | 1500 |

</div>

What this run suggests:

- **Precise labels, narrow coverage.** Each label is written from only the five articles nearest
  the centroid, so it describes the center of the cluster, not all of it. Cluster 2 is mostly
  Sci/Tech (204 of 280 by the gold labels), but only 4% of its articles mention oceans or dead
  zones; Laya keeps 73 and sends 184 to *Other*. Likewise, a quarter of cluster 4 mentions Google
  or an IPO, against 81% of the 52 articles Laya keeps.
- **Clear themes hold up.** Baseball (82%) and oil prices and stocks (88%) are confirmed for most of
  their articles.
- **Neighbouring labels compete.** Cluster 6 is about the Olympics (81% of its articles mention
  them), but its label is about swimming, which only 28% mention. Laya moves 57 of its articles to
  the baseball label, whose description ("playoff results, tour victories") reads as general
  sports.
- **More *Other* than with keywords.** 637 articles (42%) match no label, against 449 (30%) in the
  [BERTopic example](bertopic_laya.md#reading-the-results). The two runs cluster differently
  (KMeans on the raw embeddings here, UMAP + KMeans there), so this is not a like-for-like
  comparison of the labeling methods.

The trade-off is the one described in [How It Works](../algorithm.md#reading-the-comparison): a
precise, checkable label confirms a tight core and flags the rest. To cover more of each cluster,
feed `reduce` more than the five documents nearest the centroid, for example a larger sample of
[compressed](../getting_started/compression.md) documents.

!!! note
    Small models do not always follow the finer prompt rules: "Baseball Game Outcomes and
    Performances" joins two ideas with "and", which the `classes` prompt forbids. See
    [Prompt Templates](../getting_started/prompts.md).

## Full script

??? example "examples/kmeans_genlbs_laya.py"

    ```python
    --8<-- "examples/kmeans_genlbs_laya.py"
    ```
