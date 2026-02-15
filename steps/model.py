from zenml import step
import polars as pl
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

@step
def build_baskets(df_path: str) -> pl.DataFrame:
    df = pl.scan_parquet(df_path)

    baskets = (
        df
        .drop_nulls(["orderid","productid"])
        .select(["orderid", "productid"])
        .group_by("orderid")
        .agg(pl.col("productid").alias("basket"))
        # optional: nur Baskets mit >=2 Items
        .filter(pl.col("basket").list.len() >= 2)
        .collect()
    )

    #print(baskets.head())
    return baskets

@step
def data_split(baskets: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    split_idx = int(len(baskets) * 0.8)
    train_df = baskets[:split_idx]
    test_df = baskets[split_idx:]
    return train_df, test_df

@step
def train_model(train_df: pl.DataFrame) -> str:
    sentences = PolarsBasketIterator(train_df)
    model = Word2Vec(
        sentences=sentences,
        vector_size=32,
        window=100,
        sg=1,
        shrink_windows=False,
        workers=max(1, os.cpu_count() - 1),
        min_count=2
    )

    model_path = "models/word2vec.model"
    model.save(model_path)
    return model_path

class PolarsBasketIterator:
    def __init__(self, df):
        self.df = df

    def __iter__(self):
        for row in self.df.iter_rows(named=True):
            yield row["basket"]

@step
def plot_all_items_2d(model_path, max_labels=60, random_state=43):
    model = Word2Vec.load(model_path)
    items = list(model.wv.index_to_key)
    X = np.array([model.wv[pid] for pid in items])

    reducer = TSNE(n_components=2, perplexity=30, init="pca", random_state=random_state)
    X2 = reducer.fit_transform(X)

    plt.figure(figsize=(12, 9))
    plt.scatter(X2[:, 0], X2[:, 1])

    # label only a subset to avoid unreadable mess
    if len(items) <= max_labels:
        label_idx = range(len(items))
    else:
        label_idx = np.random.RandomState(random_state).choice(len(items), size=max_labels, replace=False)

    #for i in label_idx:
    #    name = prod_name.get(items[i], "UNKNOWN")
    #    label = f"{name}"   # oder nur name
    #    plt.text(X2[i, 0], X2[i, 1], label, fontsize=6)

    plt.title("t-SNE 2D projection of all product embeddings")
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.savefig("word2vec.png")

@step
def test_model(model: Word2Vec):
    print(retrieve_candidates(model, "000051-007"))

def retrieve_candidates(model: Word2Vec, anchor_pid: str, topk: int = 5):
    anchor_pid = str(anchor_pid)
    if anchor_pid not in model.wv:
        return []
    return [(pid, float(sim)) for pid, sim in model.wv.most_similar(anchor_pid, topn=topk)]