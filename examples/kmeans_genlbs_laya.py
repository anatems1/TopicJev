import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from sklearn.cluster import KMeans
from topicjev.entail import LayaEntail
from topicjev.gen import LocalEmbedder, LocalGenerator
from topicjev.prompts import load_prompt

load_dotenv()

# Dataset and model configurations
DATASET_NAME: str = "fancyzhx/ag_news"
N_SAMPLES: int = 1500
N_TOPICS: int = 7
RND_SEED: int = 42
EMBED_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
EMBED_PREFIX: str = "clustering: "
LLM_MODEL: str = "Qwen/Qwen3-1.7B"
JEV_MODEL: str = "convaiinnovations/laya-typed-decisions"

# Output directory keyed by CONFIG for reproducible caching
CONFIG = {
    "dataset": DATASET_NAME,
    "n_samples": N_SAMPLES,
    "embed_model": EMBED_MODEL,
    "embed_prefix": EMBED_PREFIX,
    "llm_model": LLM_MODEL,
    "jev_model": JEV_MODEL,
    "n_topics": N_TOPICS,
    "seed": RND_SEED,
}
OUT_DIR: Path = Path("./output") / hashlib.sha1(json.dumps(CONFIG).encode()).hexdigest()[:8]
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2))
print(f"Writing results to {OUT_DIR}")

# 1. Load dataset (1500 samples from ag_news)
if not (OUT_DIR / "docs.csv").exists():
    ds = load_dataset(DATASET_NAME, split=f"train[:{N_SAMPLES}]")
    docs_df = ds.to_pandas()
    docs_df["id"] = range(len(docs_df))
    docs_df.to_csv(OUT_DIR / "docs.csv", index=False)
docs_df = pd.read_csv(OUT_DIR / "docs.csv")
docs = docs_df["text"].tolist()

# 2. Compute dense embeddings with FAISS index
embedder = LocalEmbedder(
    EMBED_MODEL,
    prefix=EMBED_PREFIX,
    trust_remote_code=False,
    save_dir=str(OUT_DIR),
)
embeds = embedder.get_embeds(docs)

# 3. Cluster using only KMeans
if not (OUT_DIR / "doc_labels.csv").exists():
    kmeans = KMeans(n_clusters=N_TOPICS, random_state=RND_SEED)
    cluster_labels = kmeans.fit_predict(embeds)
    docs_df["topic_id"] = cluster_labels
    docs_df.to_csv(OUT_DIR / "doc_labels.csv", index=False)
docs_df = pd.read_csv(OUT_DIR / "doc_labels.csv")

# 4. Generate Topic Labels (Reduce Prompt -> Classes Prompt) with Qwen3-1.7B
if not (OUT_DIR / "topic_info.csv").exists():
    # Derive cluster centroids directly from embeddings and assigned topic labels
    centroids = np.array([
        embeds[docs_df["topic_id"] == k].mean(axis=0)
        for k in range(N_TOPICS)
    ])

    # L2-normalize centroids so inner product equals cosine similarity
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_centroids = (centroids / norms).astype(np.float32)
    _, top_5_indices = embedder.index.search(normalized_centroids, 5)

    # Free embedding model from memory before loading LLM and entailment models
    embedder.close()

    reduce_tpl = load_prompt("reduce")
    classes_tpl = load_prompt("classes")

    gen = LocalGenerator(
        LLM_MODEL,
        json_mode=True,
        temperature=0.1,
        max_tokens=512,
    )

    topic_records = []
    used_titles: list[str] = []

    with gen:
        for topic_id in range(N_TOPICS):
            # Pull top 5 representative documents from FAISS
            top_docs = [docs[idx] for idx in top_5_indices[topic_id]]
            facts_json = json.dumps({"facts": top_docs}, indent=2)

            # Step 4a: Run reduce prompt to synthesize dominant theme
            print(f"\n[Topic {topic_id}] Generating synthesis via reduce prompt...")
            reduce_results = gen.batch(
                reduce_tpl,
                [{"text": facts_json}],
                req_keys=["title", "summary", "keywords"],
            )
            reduce_out = reduce_results[0]
            print(f"  Reduce Title:   {reduce_out.get('title')}")
            print(f"  Keywords:       {reduce_out.get('keywords')}")

            # Step 4b: Pass reduce output into classes prompt to derive an entailment-checkable class
            print(f"[Topic {topic_id}] Generating checkable class label via classes prompt...")
            input_json = json.dumps(reduce_out, indent=2)
            used_titles_str = json.dumps(used_titles) if used_titles else "None"

            class_results = gen.batch(
                classes_tpl,
                [{"text": input_json, "used_titles": used_titles_str}],
                req_keys=["title", "desc"],
            )
            class_out = class_results[0]
            final_title = class_out.get("title", reduce_out.get("title", f"Topic_{topic_id}"))
            final_desc = class_out.get("desc", reduce_out.get("summary", ""))
            used_titles.append(final_title)

            print(f"  Class Title:    {final_title}")
            print(f"  Description:    {final_desc}")

            topic_records.append({
                "Topic": topic_id,
                "Count": int((docs_df["topic_id"] == topic_id).sum()),
                "Name": final_title,
                "Description": final_desc,
                "Keywords": json.dumps(reduce_out.get("keywords", [])),
                "Reduce_Title": reduce_out.get("title", ""),
                "Summary": reduce_out.get("summary", ""),
                "Representative_Docs": json.dumps(top_docs),
            })

    tinfo = pd.DataFrame(topic_records)
    tinfo.to_csv(OUT_DIR / "topic_info.csv", index=False)
else:
    # Free embedding model from memory if topic_info is already cached
    embedder.close()

tinfo = pd.read_csv(OUT_DIR / "topic_info.csv")

print("\n--- Discovered Topics ---")
for _, row in tinfo.iterrows():
    print(f"Topic {row['Topic']} (n={row['Count']}): {row['Name']} - {row['Description']}")

# 5. Bootstrap Entailment with Laya (JevLike model)
if not (OUT_DIR / "jev_labels.csv").exists():
    jev = LayaEntail(JEV_MODEL)
    with jev:
        # Formulate criteria schema mapping each topic name to its description & keywords
        topic_criteria = []
        for _, row in tinfo.iterrows():
            kw_list = json.loads(row["Keywords"]) if isinstance(row["Keywords"], str) else []
            criteria_text = f"{row['Description']} (Keywords: {', '.join(kw_list)})" if kw_list else row["Description"]
            topic_criteria.append(json.dumps({row["Name"]: criteria_text}))

        results = jev.entail(docs=docs, lbls=topic_criteria)

    docs_df["jev"] = [r["class"] for r in results]
    docs_df.to_csv(OUT_DIR / "jev_labels.csv", index=False)
docs_df = pd.read_csv(OUT_DIR / "jev_labels.csv")

# 6. Calculate agreement between KMeans clusters and Laya entailment decisions
print("\n--- Topic Alignment & Agreement ---")
for topic_id in sorted(docs_df["topic_id"].unique()):
    topic_docs = docs_df[docs_df["topic_id"] == topic_id]
    agreement = (topic_docs["topic_id"] == topic_docs["jev"]).mean()
    topic_name = tinfo.loc[tinfo["Topic"] == topic_id, "Name"].values[0]
    print(f"Topic {topic_id} [{topic_name}]: Agreement between KMeans and Laya: {agreement:.2%}")
