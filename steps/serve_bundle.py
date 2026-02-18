import numpy as np
import polars as pl
import lightgbm as lgb
from gensim.models import Word2Vec


# -----------------------
# Load artifacts
# -----------------------

def load_models(w2v_path: str, lgbm_path: str):
    w2v = Word2Vec.load(w2v_path)
    ranker = lgb.Booster(model_file=lgbm_path)  # wenn du Booster gespeichert hast
    return w2v, ranker


def build_lookup_dicts(
    commerces_path: str,
    products_path: str,
    pop_global_path: str,
    pop_store_path: str,
    pop_region_path: str,
    pop_subch_path: str,
    pop_origin_path: str,
):
    commerces = pl.read_parquet(commerces_path).select(
        ["userid", "region", "subchannel", "channel", "commune"]
    )
    products = pl.read_parquet(products_path).select(
        ["productid", "category", "blocked"]
    )

    # store / product meta
    store_meta = {
        r["userid"]: {
            "region": r["region"],
            "subchannel": r["subchannel"],
            "channel": r["channel"],
            "commune": r["commune"],
        }
        for r in commerces.iter_rows(named=True)
    }

    prod_cat = {r["productid"]: r["category"] for r in products.iter_rows(named=True)}
    prod_blocked = {r["productid"]: bool(r["blocked"]) for r in products.iter_rows(named=True)}

    # popularity lookup dicts
    pop_global = {
        r["productid"]: int(r["pop_global"])
        for r in pl.read_parquet(pop_global_path).iter_rows(named=True)
    }

    pop_store = {
        (r["userid"], r["productid"]): int(r["pop_store"])
        for r in pl.read_parquet(pop_store_path).iter_rows(named=True)
    }

    pop_region = {
        (r["region"], r["productid"]): int(r["pop_region"])
        for r in pl.read_parquet(pop_region_path).iter_rows(named=True)
    }

    pop_subch = {
        (r["subchannel"], r["productid"]): int(r["pop_subch"])
        for r in pl.read_parquet(pop_subch_path).iter_rows(named=True)
    }

    pop_origin = {
        (r["origin"], r["productid"]): int(r["pop_origin"])
        for r in pl.read_parquet(pop_origin_path).iter_rows(named=True)
    }

    return store_meta, prod_cat, prod_blocked, pop_global, pop_store, pop_region, pop_subch, pop_origin


# -----------------------
# Inference
# -----------------------

def recommend_candidates(
    anchor: str,
    userid: str,
    origin: str,
    w2v: Word2Vec,
    ranker: lgb.Booster,
    store_meta: dict,
    prod_cat: dict,
    prod_blocked: dict,
    pop_global: dict,
    pop_store: dict,
    pop_region: dict,
    pop_subch: dict,
    pop_origin: dict,
    topk_retrieval: int = 50,
    topn: int = 10,
    basket: set[str] | None = None,
):
    basket = basket or set()

    if anchor not in w2v.wv:
        return []

    ctx = store_meta.get(userid, {})
    region = ctx.get("region", "UNKNOWN")
    subch = ctx.get("subchannel", "UNKNOWN")

    # 1) retrieve
    retrieved = w2v.wv.most_similar(anchor, topn=topk_retrieval)

    # 2) build candidate list with rank
    candidates = []
    for rank, (cand, sim) in enumerate(retrieved, start=1):
        if cand == anchor:
            continue
        if cand in basket:
            continue
        if prod_blocked.get(cand, False):
            continue
        if cand not in w2v.wv:
            continue

        candidates.append((cand, rank, float(sim)))

    if not candidates:
        return []

    # 3) feature matrix
    anchor_cat = prod_cat.get(anchor, "UNKNOWN")

    rows = []
    for cand, r_rank, w2v_sim in candidates:
        cand_cat = prod_cat.get(cand, "UNKNOWN")
        same_cat = 1 if cand_cat == anchor_cat else 0

        pg = pop_global.get(cand, 0)
        ps = pop_store.get((userid, cand), 0)
        pr = pop_region.get((region, cand), 0)
        psub = pop_subch.get((subch, cand), 0)
        po = pop_origin.get((origin, cand), 0)

        # optional transforms (recommended)
        pg = np.log1p(pg)
        ps = np.log1p(ps)
        pr = np.log1p(pr)
        psub = np.log1p(psub)
        po = np.log1p(po)

        rows.append([
            w2v_sim,           # sim_item2vec
            same_cat,          # same_category
            pg, ps, pr, psub, po
            # + retrieval_rank feature, if you trained with it
            # r_rank
        ])

    X = np.asarray(rows, dtype=np.float32)

    # 4) predict scores
    scores = ranker.predict(X)

    # 5) sort
    ranked = sorted(
        zip([c[0] for c in candidates], scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked[:topn]


# -----------------------
# Example usage
# -----------------------

if __name__ == "__main__":
    w2v_path = "models/word2vec.model"
    lgbm_path = "models/lgbm_ranker.txt"

    w2v, ranker = load_models(w2v_path, lgbm_path)

    store_meta, prod_cat, prod_blocked, pop_global, pop_store, pop_region, pop_subch, pop_origin = build_lookup_dicts(
        commerces_path="data/commerces.parquet",
        products_path="data/products_v2.parquet",
        pop_global_path="artifacts/pop_global.parquet",
        pop_store_path="artifacts/pop_store.parquet",
        pop_region_path="artifacts/pop_region.parquet",
        pop_subch_path="artifacts/pop_subch.parquet",
        pop_origin_path="artifacts/pop_origin.parquet",
    )

    recs = recommend_candidates(
        anchor="000006-001",
        userid="12345",
        origin="RUTA",
        w2v=w2v,
        ranker=ranker,
        store_meta=store_meta,
        prod_cat=prod_cat,
        prod_blocked=prod_blocked,
        pop_global=pop_global,
        pop_store=pop_store,
        pop_region=pop_region,
        pop_subch=pop_subch,
        pop_origin=pop_origin,
        topk_retrieval=50,
        topn=10,
        basket=set(),  # falls du gerade einen Warenkorb hast, hier rein
    )

    print(recs)