import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from gensim.models import Word2Vec
from tqdm.auto import tqdm
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

_W2V = None
_RANKER = None
_STORE_META = None
_PROD_CAT = None
_POP_GLOBAL = None
_POP_STORE = None
_POP_REGION = None
_POP_SUBCH = None
_PROD_NAME = None
# -----------------------
# Load artifacts
# -----------------------

FEATURE_COLS = [
    "sim_item2vec",
    "pop_global",
    "pop_subch",
    #"pop_origin",
    "pop_region",
    "channel",
    "pop_store",
    "commune",
    "cand_category",
    #"origin",
    "region",
    "subchannel",
]

CATEGORICAL_COLS = [
    "channel",
    "commune",
    "cand_category",
    #"origin",
    "region",
    "subchannel",
]

def get_runtime_objects():
    global _W2V, _RANKER
    global _STORE_META, _PROD_CAT, _POP_GLOBAL, _POP_STORE, _POP_REGION, _POP_SUBCH
    global _PROD_NAME

    if _W2V is None or _RANKER is None:
        w2v_path = str(MODELS_DIR / "word2vec.model")
        lgbm_path = str(MODELS_DIR / "lgbm_ranker.txt")
        _W2V, _RANKER = load_models(w2v_path, lgbm_path)

    if (
        _STORE_META is None
        or _PROD_CAT is None
        or _POP_GLOBAL is None
        or _POP_STORE is None
        or _POP_REGION is None
        or _POP_SUBCH is None
    ):
        (
            _STORE_META,
            _PROD_CAT,
            _POP_GLOBAL,
            _POP_STORE,
            _POP_REGION,
            _POP_SUBCH,
        ) = build_lookup_dicts(
            commerces_path=str(DATA_DIR / "commerces.parquet"),
            products_path=str(DATA_DIR / "products_v2.parquet"),
            pop_global_path=str(ARTIFACTS_DIR / "pop_global.parquet"),
            pop_store_path=str(ARTIFACTS_DIR / "pop_store.parquet"),
            pop_region_path=str(ARTIFACTS_DIR / "pop_region.parquet"),
            pop_subch_path=str(ARTIFACTS_DIR / "pop_subch.parquet"),
        )

    if _PROD_NAME is None:
        _PROD_NAME = {
            _to_key(r["productid"]): _to_key(r["name"])
            for r in pl.read_parquet(str(DATA_DIR / "products_v2.parquet"))
            .select(["productid", "name"])
            .iter_rows(named=True)
        }

    return (
        _W2V,
        _RANKER,
        _STORE_META,
        _PROD_CAT,
        _POP_GLOBAL,
        _POP_STORE,
        _POP_REGION,
        _POP_SUBCH,
        _PROD_NAME,
    )

def _to_key(v) -> str:
    return str(v)

def load_models(w2v_path: str, lgbm_path: str):
    w2v = Word2Vec.load(w2v_path)
    ranker = lgb.Booster(model_file=lgbm_path)
    return w2v, ranker

def build_lookup_dicts(
    commerces_path: str,
    products_path: str,
    pop_global_path: str,
    pop_store_path: str,
    pop_region_path: str,
    pop_subch_path: str,
    #pop_origin_path: str,
):
    commerces = pl.read_parquet(commerces_path).select(
        ["userid", "region", "subchannel", "channel", "commune"]
    )
    products = pl.read_parquet(products_path).select(
        ["productid", "category", "blocked"]
    )

    # store / product meta
    store_meta = {
        _to_key(r["userid"]): {
            "region": r["region"],
            "subchannel": r["subchannel"],
            "channel": r["channel"],
            "commune": r["commune"],
        }
        for r in commerces.iter_rows(named=True)
    }

    prod_cat = {_to_key(r["productid"]): r["category"] for r in products.iter_rows(named=True)}
    #prod_blocked = {_to_key(r["productid"]): bool(r["blocked"]) for r in products.iter_rows(named=True)}

    # popularity lookup dicts
    pop_global = {
        _to_key(r["productid"]): int(r["pop_global"])
        for r in pl.read_parquet(pop_global_path).iter_rows(named=True)
    }

    pop_store = {
        (_to_key(r["userid"]), _to_key(r["productid"])): int(r["pop_store"])
        for r in pl.read_parquet(pop_store_path).iter_rows(named=True)
    }

    pop_region = {
        (_to_key(r["region"]), _to_key(r["productid"])): int(r["pop_region"])
        for r in pl.read_parquet(pop_region_path).iter_rows(named=True)
    }

    pop_subch = {
        (_to_key(r["subchannel"]), _to_key(r["productid"])): int(r["pop_subch"])
        for r in pl.read_parquet(pop_subch_path).iter_rows(named=True)
    }
    """
    pop_origin = {
        (_to_key(r["origin"]), _to_key(r["productid"])): int(r["pop_origin"])
        for r in pl.read_parquet(pop_origin_path).iter_rows(named=True)
    }
    """
    #return store_meta, prod_cat, prod_blocked, pop_global, pop_store, pop_region, pop_subch, pop_origin
    return store_meta, prod_cat, pop_global, pop_store, pop_region, pop_subch


# -----------------------
# Inference
# -----------------------

def recommend_candidates(
    anchor: str,
    userid: str,
    #origin: str,
    w2v: Word2Vec,
    ranker: lgb.Booster,
    store_meta: dict,
    prod_cat: dict,
    #prod_blocked: dict,
    pop_global: dict,
    pop_store: dict,
    pop_region: dict,
    pop_subch: dict,
    #pop_origin: dict,
    topn: int = 10,
    basket: set[str] | None = None,
):
    basket = {_to_key(x) for x in (basket or set())}
    anchor = _to_key(anchor)
    userid = _to_key(userid)
    #origin = _to_key(origin)

    retrieved = []
    if anchor in w2v.wv:
        retrieved = w2v.wv.most_similar(anchor, topn=max(50, topn))
    else:
        fallback_n = max(0, 2 * int(topn))
        if fallback_n == 0:
            return []

        for rank, (cand, _) in enumerate(
            sorted(pop_global.items(), key=lambda x: x[1], reverse=True),
            start=1,
        ):
            cand = _to_key(cand)
            if cand == anchor:
                continue
            if cand in basket:
                continue
            retrieved.append((rank, (cand, 0.0)))
            if len(retrieved) >= fallback_n:
                break

        if not retrieved:
            return []

    ctx = store_meta.get(userid, {})
    region = _to_key(ctx.get("region", "UNKNOWN"))
    subch = _to_key(ctx.get("subchannel", "UNKNOWN"))
    channel = _to_key(ctx.get("channel", "UNKNOWN"))
    commune = _to_key(ctx.get("commune", "UNKNOWN"))

    # 1) retrieve
    #retrieved = w2v.wv.most_similar(anchor, topn=max(50, topn))

    # 2) build candidate list with rank
    candidates = []
    for rank, (cand, sim) in enumerate(retrieved, start=1):
        cand = _to_key(cand)
        if cand == anchor:
            continue
        if cand in basket:
            continue
        #if prod_blocked.get(cand, False):
        #    continue
        if cand not in w2v.wv:
            continue

        candidates.append((cand, rank, float(sim)))

    if not candidates:
        fallback_n = max(0, 2 * int(topn))
        if fallback_n == 0:
            return []

        for rank, (cand, _) in enumerate(
            sorted(pop_global.items(), key=lambda x: x[1], reverse=True),
            start=1,
        ):
            cand = _to_key(cand)
            if cand == anchor:
                continue
            if cand in basket:
                continue
            candidates.append((cand, rank, 0.0))
            if len(candidates) >= fallback_n:
                break

        if not candidates:
            return []

    rows = []
    for cand, r_rank, w2v_sim in candidates:
        cand_cat = _to_key(prod_cat.get(cand, "UNKNOWN"))

        pg = pop_global.get(cand, 0)
        ps = pop_store.get((userid, cand), 0)
        pr = pop_region.get((region, cand), 0)
        psub = pop_subch.get((subch, cand), 0)
        #po = pop_origin.get((origin, cand), 0)

        pg = np.log1p(pg)
        ps = np.log1p(ps)
        pr = np.log1p(pr)
        psub = np.log1p(psub)
        #po = np.log1p(po)

        rows.append(
            {
                "sim_item2vec": w2v_sim,
                "pop_global": pg,
                "pop_subch": psub,
                #"pop_origin": po,
                "pop_region": pr,
                "channel": channel,
                "pop_store": ps,
                "commune": commune,
                "cand_category": cand_cat,
                #"origin": origin,
                "region": region,
                "subchannel": subch,
            }
        )

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    for col in CATEGORICAL_COLS:
        X[col] = X[col].fillna("UNKNOWN").astype("category")
    for col in FEATURE_COLS:
        if col not in CATEGORICAL_COLS:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(np.float32)

    # 4) predict scores
    scores = ranker.predict(X)

    # 5) sort
    ranked = sorted(
        zip([c[0] for c in candidates], scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked[:topn]

def getMultiRec(anchors_df: pl.DataFrame) -> pl.DataFrame:
    """
    anchors_df must contains the colums "anchor_pid" and "userid".
    """
    w2v_path = str(REPO_ROOT / "models" / "word2vec.model")
    lgbm_path = str(REPO_ROOT / "models" / "lgbm_ranker.txt")
    topn = 20
    #anchors_df = anchors_df.drop("origin")

    w2v, ranker = load_models(w2v_path, lgbm_path)

    store_meta, prod_cat, pop_global, pop_store, pop_region, pop_subch = build_lookup_dicts(
        commerces_path=str(REPO_ROOT / "data" / "commerces.parquet"),
        products_path=str(REPO_ROOT / "data" / "products_v2.parquet"),
        pop_global_path=str(REPO_ROOT / "artifacts" / "pop_global.parquet"),
        pop_store_path=str(REPO_ROOT / "artifacts" / "pop_store.parquet"),
        pop_region_path=str(REPO_ROOT / "artifacts" / "pop_region.parquet"),
        pop_subch_path=str(REPO_ROOT / "artifacts" / "pop_subch.parquet"),
        #pop_origin_path="artifacts/pop_origin.parquet",
    )

    unknown = "UNKNOWN"

    unique_anchors = {_to_key(k) for k in w2v.wv.key_to_index.keys()}
    retrieval_cache: dict[str, list[tuple[str, float]]] = {}
    for anchor in unique_anchors:
        retrieval_cache[anchor] = [
            (cand, sim)
            for cand, sim in w2v.wv.most_similar(anchor, topn=50)
        ]

    results = []
    for anchor_pid, userid in tqdm(
        anchors_df.iter_rows(),
        total=anchors_df.height,
        desc="Generate recommendations",
    ):
        retrieved = retrieval_cache.get(anchor_pid, [])
        if not retrieved:
            fallback_n = max(0, 2 * int(topn))
            if fallback_n == 0:
                return []

            for rank, (cand, _) in enumerate(
                sorted(pop_global.items(), key=lambda x: x[1], reverse=True),
                start=1,
            ):
                cand = _to_key(cand)
                if cand == anchor_pid:
                    continue
                retrieved.append((cand, 0.0))
                if len(retrieved) >= fallback_n:
                    break

            if not retrieved:
                results.append(
                {
                    "anchor_id": anchor_pid,
                    "kiosk_id": userid,
                    "recs": [],
                }
                )
                continue

        ctx = store_meta.get(userid, {})
        region = ctx.get("region", unknown)
        subch = ctx.get("subchannel", unknown)
        channel = ctx.get("channel", unknown)
        commune = ctx.get("commune", unknown)

        candidates: list[str] = []
        sim_item2vec: list[float] = []
        pop_global_vals: list[float] = []
        pop_store_vals: list[float] = []
        pop_region_vals: list[float] = []
        pop_subch_vals: list[float] = []
        #pop_origin_vals: list[float] = []
        cand_cats: list[str] = []

        for rank, (cand, sim) in enumerate(retrieved, start=1):
            candidates.append(cand)
            sim_item2vec.append(sim)
            pop_global_vals.append(float(np.log1p(pop_global.get(cand, 0))))
            pop_store_vals.append(float(np.log1p(pop_store.get((userid, cand), 0))))
            pop_region_vals.append(float(np.log1p(pop_region.get((region, cand), 0))))
            pop_subch_vals.append(float(np.log1p(pop_subch.get((subch, cand), 0))))
            #pop_origin_vals.append(float(np.log1p(pop_origin.get((origin, cand), 0))))
            cand_cats.append(prod_cat.get(cand, unknown))

        n_candidates = len(candidates)
        if n_candidates == 0:
            results.append(
                {
                    "anchor_id": anchor_pid,
                    "userid": userid,
                    #"origin": origin,
                    "recs": [],
                }
            )
            continue

        X = pd.DataFrame(
            {
                "sim_item2vec": np.asarray(sim_item2vec, dtype=np.float32),
                "pop_global": np.asarray(pop_global_vals, dtype=np.float32),
                "pop_subch": np.asarray(pop_subch_vals, dtype=np.float32),
                #"pop_origin": np.asarray(pop_origin_vals, dtype=np.float32),
                "pop_region": np.asarray(pop_region_vals, dtype=np.float32),
                "channel": pd.Categorical([channel] * n_candidates),
                "pop_store": np.asarray(pop_store_vals, dtype=np.float32),
                "commune": pd.Categorical([commune] * n_candidates),
                "cand_category": pd.Categorical(cand_cats),
                #"origin": pd.Categorical([origin] * n_candidates),
                "region": pd.Categorical([region] * n_candidates),
                "subchannel": pd.Categorical([subch] * n_candidates),
            },
            columns=FEATURE_COLS,
        )

        scores = np.asarray(ranker.predict(X), dtype=np.float32)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        recs = ranked[:topn]
        recs_to_add = [_to_key(pid) for pid, _ in recs]

        results.append(
            {
                "anchor_id": anchor_pid,
                "userid": userid,
                #"origin": origin,
                "recs": recs_to_add,
            }
        )

    return pl.DataFrame(results, schema={
        "anchor_id": pl.Utf8,
        "userid": pl.Utf8,
        #"origin": pl.Utf8,
        "recs": pl.List(pl.Utf8)
        })

def getSingleRec(anchor_id: str, user_id: str, topn: int = 30, addDebugInfo: bool = False) -> pl.DataFrame:
    (
        w2v,
        ranker,
        store_meta,
        prod_cat,
        pop_global,
        pop_store,
        pop_region,
        pop_subch,
        prod_name,
    ) = get_runtime_objects()

    recs = recommend_candidates(
        anchor=anchor_id,
        userid=user_id,
        #origin=origin,
        w2v=w2v,
        ranker=ranker,
        store_meta=store_meta,
        prod_cat=prod_cat,
        pop_global=pop_global,
        pop_store=pop_store,
        pop_region=pop_region,
        pop_subch=pop_subch,
        #pop_origin=pop_origin,
        topn=topn,
        basket=set(),  # falls gerade einen Warenkorb besteht, hier rein
    )

    anchor_name = prod_name.get(anchor_id, "UNKNOWN")
    print(f"Eingabeprodukt: {anchor_id} | {anchor_name}")

    recs_with_name = [
        {
            "productid": pid,
            "name": prod_name.get(_to_key(pid), "UNKNOWN"),
            "score": float(score),
        }
        for pid, score in recs
    ]
    recs_df = pd.DataFrame(recs_with_name, columns=["productid", "name", "score"])
    if recs_df.empty:
        print("Keine Empfehlungen gefunden.")
    else:
        print(recs_df.to_string(index=False))

    
    return_df = (
        pl.DataFrame(
            {
                "anchor_id": anchor_id,
                "user_id": user_id,
                "product_id": [_to_key(pid) for pid, _ in recs],
                "score": [float(score) for _, score in recs],
            }
        )
        .with_columns(
            pl.col("product_id")
            .map_elements(lambda pid: prod_name.get(pid, "UNKNOWN"), return_dtype=pl.Utf8)
            .alias("name")
        )
        .select(
            ["anchor_id", "user_id", "product_id"] + (["name", "score"] if addDebugInfo else [])
        )
    )


    return return_df

if __name__ == "__main__":
    """
    input_df = pl.scan_csv("data/2024-20250001_part_00-001_short.csv")
    input_df = (
        input_df
        .rename({"productid": "anchor_pid"})
        .select(["anchor_pid", "userid"])
        .unique(pl.col("userid"))
    )
    results = getMultiRec(input_df.collect())
    print(results)
    """
    
    print(getSingleRec("000295-999", "9077130ee9894b2d1e6d3341b341e006", topn=8, addDebugInfo = False))
    
    data = {"anchor_pid": ["000295-003"], "userid": ["9077130ee9894b2d1e6d3341b341e006"]}
    print(getMultiRec(pl.DataFrame(data, schema={"anchor_pid": str, "userid": str})))
