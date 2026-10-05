import ast
import json
import hashlib
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from topicjev.entail import LayaEntail
from topicjev.gen import LocalEmbedder

load_dotenv()
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
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2))
print(f"Writing results to {OUT_DIR}")

# establish topic model for bootstrap
from umap import UMAP
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance

tmodel = BERTopic(
    nr_topics=N_TOPICS,
    # BERTopic's default UMAP, seeded for reproducible runs
    umap_model=UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RND_SEED,
    ),
    hdbscan_model=KMeans(n_clusters=N_TOPICS, random_state=RND_SEED),
    vectorizer_model=CountVectorizer(
        max_df=0.8,
        stop_words="english",
        ngram_range=(1, 2),
    ),
    ctfidf_model=ClassTfidfTransformer(
        bm25_weighting=True,
        reduce_frequent_words=True,
    ),
    representation_model=MaximalMarginalRelevance(diversity=0.3),
    verbose=False,
)


# load dataset (1500 samples from ag_news) and generate embeddings
if not (OUT_DIR / "docs.csv").exists():
    ds = load_dataset(DATASET_NAME, split=f"train[:{N_SAMPLES}]")
    docs_df = ds.to_pandas()
    docs_df["id"] = range(len(docs_df))
    docs_df.to_csv(OUT_DIR / "docs.csv", index=False)
docs_df = pd.read_csv(OUT_DIR / "docs.csv")
docs = docs_df["text"].tolist()
# use transformers' built-in NomicBert; the checkpoint's remote code predates transformers 5
embedder = LocalEmbedder(
    EMBED_MODEL,
    prefix=EMBED_PREFIX,
    trust_remote_code=False,
    save_dir=str(OUT_DIR),
)
embeds = embedder.get_embeds(docs)

# run topic model
if not (OUT_DIR / "doc_labels.csv").exists():
    tmodel.fit_transform(docs, embeddings=embeds)
    tinfo = tmodel.get_topic_info()
    tinfo.to_csv(OUT_DIR / "topic_info.csv", index=False)
    docs_df["topic_id"] = tmodel.topics_
    docs_df.to_csv(OUT_DIR / "doc_labels.csv", index=False)
docs_df = pd.read_csv(OUT_DIR / "doc_labels.csv")
tinfo = pd.read_csv(OUT_DIR / "topic_info.csv")


# BOOTSTRAP with Laya (JevLike model)
if not (OUT_DIR / "jev_labels.csv").exists():
    jev = LayaEntail(JEV_MODEL)
    with jev:
        results = jev.entail(
            docs=docs,
            lbls=[
                json.dumps(
                    {
                        row["Name"]: ", ".join(
                            ast.literal_eval(row["Representation"])[:5]
                        )
                    }
                )
                for _, row in tinfo.iterrows()
            ],
        )
    docs_df["jev"] = [r["class"] for r in results]
    docs_df.to_csv(OUT_DIR / "jev_labels.csv", index=False)
docs_df = pd.read_csv(OUT_DIR / "jev_labels.csv")

# calculate agreement
for topic_id in docs_df["topic_id"].unique():
    topic_docs = docs_df[docs_df["topic_id"] == topic_id]
    agreement = (topic_docs["topic_id"] == topic_docs["jev"]).mean()
    print(f"Topic {topic_id}: Agreement between BERTopic and Laya: {agreement:.2%}")
