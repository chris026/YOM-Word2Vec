import argparse
from typing import List, Tuple

import polars as pl

from serve_bundle import (
    build_lookup_dicts,
    load_models,
    recommend_candidates,
)


def parse_query(raw: str) -> Tuple[str, str, str]:
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) != 3:
        raise ValueError(
            f"Ungueltiges Query-Format '{raw}'. Erwartet: anchor|userid|origin"
        )
    return parts[0], parts[1], parts[2]


def default_queries_from_orders(orders_path: str, limit: int) -> List[Tuple[str, str, str]]:
    df = (
        pl.read_parquet(orders_path)
        .select(["productid", "userid", "origin"])
        .drop_nulls(["productid", "userid", "origin"])
        .unique()
        .head(limit)
    )
    return [
        (str(r["productid"]), str(r["userid"]), str(r["origin"]))
        for r in df.iter_rows(named=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Testbench fuer serve_bundle Empfehlungen")
    parser.add_argument("--w2v-path", default="models/word2vec.model")
    parser.add_argument("--lgbm-path", default="models/lgbm_ranker.txt")
    parser.add_argument("--commerces-path", default="data/commerces.parquet")
    parser.add_argument("--products-path", default="data/products_v2.parquet")
    parser.add_argument("--pop-global-path", default="artifacts/pop_global.parquet")
    parser.add_argument("--pop-store-path", default="artifacts/pop_store.parquet")
    parser.add_argument("--pop-region-path", default="artifacts/pop_region.parquet")
    parser.add_argument("--pop-subch-path", default="artifacts/pop_subch.parquet")
    parser.add_argument("--pop-origin-path", default="artifacts/pop_origin.parquet")
    parser.add_argument("--orders-path", default="data/2024-20250001_part_00-001.parquet")
    parser.add_argument("--topk-retrieval", type=int, default=50)
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--num-default-queries", type=int, default=5)
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Format: anchor|userid|origin; kann mehrfach gesetzt werden",
    )
    args = parser.parse_args()

    w2v, ranker = load_models(args.w2v_path, args.lgbm_path)
    (
        store_meta,
        prod_cat,
        #prod_blocked,
        pop_global,
        pop_store,
        pop_region,
        pop_subch,
        pop_origin,
    ) = build_lookup_dicts(
        commerces_path=args.commerces_path,
        products_path=args.products_path,
        pop_global_path=args.pop_global_path,
        pop_store_path=args.pop_store_path,
        pop_region_path=args.pop_region_path,
        pop_subch_path=args.pop_subch_path,
        pop_origin_path=args.pop_origin_path,
    )

    if args.query:
        queries = [parse_query(q) for q in args.query]
    else:
        queries = default_queries_from_orders(
            orders_path=args.orders_path, limit=args.num_default_queries
        )

    print("serve_bundle Testbench")
    print(f"Anzahl Queries: {len(queries)}")
    print(f"topk_retrieval: {args.topk_retrieval}")
    print(f"topn: {args.topn}")

    for i, (anchor, userid, origin) in enumerate(queries, start=1):
        print(f"\n=== Query {i} ===")
        print(f"anchor={anchor} | userid={userid} | origin={origin}")

        recs = recommend_candidates(
            anchor=anchor,
            userid=userid,
            origin=origin,
            w2v=w2v,
            ranker=ranker,
            store_meta=store_meta,
            prod_cat=prod_cat,
            #prod_blocked=prod_blocked,
            pop_global=pop_global,
            pop_store=pop_store,
            pop_region=pop_region,
            pop_subch=pop_subch,
            pop_origin=pop_origin,
            topk_retrieval=args.topk_retrieval,
            topn=args.topn,
            basket=set(),
        )

        if not recs:
            print("Keine Empfehlungen (Anchor evtl. OOV oder keine gueltigen Kandidaten).")
            continue

        for rank, (pid, score) in enumerate(recs, start=1):
            print(f"{rank:2d}. {pid}  score={float(score):.6f}")


if __name__ == "__main__":
    main()
