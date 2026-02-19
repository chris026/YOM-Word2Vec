import argparse
import csv
import random
import time

import numpy as np
from gensim.models import Word2Vec


def summarize(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def sample_cosine_distances(model: Word2Vec, num_pairs: int, seed: int) -> tuple[np.ndarray, dict]:
    rng = random.Random(seed)
    vectors = model.wv.vectors
    vocab_size = vectors.shape[0]

    if vocab_size < 2:
        raise ValueError("Vokabular hat weniger als 2 Produkte.")

    num_pairs = min(num_pairs, vocab_size * (vocab_size - 1) // 2)
    distances = np.empty(num_pairs, dtype=np.float64)

    norms = np.linalg.norm(vectors, axis=1)

    for i in range(num_pairs):
        a, b = rng.sample(range(vocab_size), 2)

        na = norms[a]
        nb = norms[b]

        if na == 0.0 or nb == 0.0:
            distances[i] = 1.0
            continue

        cos_sim = float(np.dot(vectors[a], vectors[b]) / (na * nb))
        cos_dist = 1.0 - cos_sim
        distances[i] = cos_dist

    return distances, summarize(distances)


def reciprocal_neighbor_rate(
    model: Word2Vec,
    topk: int = 10,
    num_anchors: int = 300,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    keys = list(model.wv.index_to_key)
    if not keys:
        raise ValueError("Modell enthaelt keine Produkte.")

    anchors = keys if len(keys) <= num_anchors else rng.sample(keys, num_anchors)

    checked_links = 0
    reciprocal_links = 0

    for anchor in anchors:
        neighbors = model.wv.most_similar(anchor, topn=topk)

        for nb, _ in neighbors:
            nb_neighbors = model.wv.most_similar(nb, topn=topk)
            nb_neighbor_ids = {pid for pid, _ in nb_neighbors}
            checked_links += 1
            if anchor in nb_neighbor_ids:
                reciprocal_links += 1

    rate = (reciprocal_links / checked_links) if checked_links else 0.0

    return {
        "anchors": len(anchors),
        "topk": topk,
        "checked_links": checked_links,
        "reciprocal_links": reciprocal_links,
        "reciprocal_rate": rate,
    }


def random_anchor_distance_extremes(
    model: Word2Vec,
    seed: int = 42,
    topn: int = 10,
) -> dict:
    rng = random.Random(seed)
    keys = list(model.wv.index_to_key)
    if len(keys) < 2:
        raise ValueError("Vokabular hat weniger als 2 Produkte.")

    anchor = rng.choice(keys)
    anchor_vec = model.wv[anchor]
    anchor_norm = np.linalg.norm(anchor_vec)

    if anchor_norm == 0.0:
        raise ValueError(f"Anchor-Vektor hat Norm 0: {anchor}")

    rows = []
    for pid in keys:
        if pid == anchor:
            continue
        vec = model.wv[pid]
        n = np.linalg.norm(vec)
        if n == 0.0:
            cos_dist = 1.0
        else:
            cos_sim = float(np.dot(anchor_vec, vec) / (anchor_norm * n))
            cos_dist = 1.0 - cos_sim
        rows.append((pid, cos_dist))

    rows_sorted = sorted(rows, key=lambda x: x[1])
    nearest = rows_sorted[:topn]
    farthest = list(reversed(rows_sorted[-topn:]))

    return {
        "anchor": anchor,
        "nearest": nearest,
        "farthest": farthest,
    }


def load_baskets_from_csv(csv_path: str, max_orders: int | None = None) -> list[list[str]]:
    by_order = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            oid = row.get("orderid")
            pid = row.get("productid")
            if not oid or not pid:
                continue
            by_order.setdefault(oid, []).append(pid)

    baskets = [items for items in by_order.values() if len(items) >= 2]
    if max_orders is not None and len(baskets) > max_orders:
        baskets = baskets[:max_orders]
    return baskets


def basket_retrieval_metrics(
    model: Word2Vec,
    baskets: list[list[str]],
    topk: int = 10,
) -> dict:
    total_anchors = 0
    used_anchors = 0
    oov_anchors = 0
    hit_count = 0
    rr_sum = 0.0

    for basket in baskets:
        basket_set = set(basket)
        for anchor in basket:
            total_anchors += 1
            if anchor not in model.wv:
                oov_anchors += 1
                continue

            positives = basket_set - {anchor}
            positives_in_vocab = {p for p in positives if p in model.wv}
            if not positives_in_vocab:
                continue

            used_anchors += 1
            retrieved = model.wv.most_similar(anchor, topn=topk)
            retrieved_ids = [pid for pid, _ in retrieved]

            if any(pid in positives_in_vocab for pid in retrieved_ids):
                hit_count += 1

            rr = 0.0
            for i, pid in enumerate(retrieved_ids, start=1):
                if pid in positives_in_vocab:
                    rr = 1.0 / i
                    break
            rr_sum += rr

    hit_rate = (hit_count / used_anchors) if used_anchors else 0.0
    mrr = (rr_sum / used_anchors) if used_anchors else 0.0
    oov_rate = (oov_anchors / total_anchors) if total_anchors else 0.0

    return {
        "num_baskets": len(baskets),
        "total_anchors": total_anchors,
        "used_anchors": used_anchors,
        "oov_anchors": oov_anchors,
        "oov_rate": oov_rate,
        "hit_rate_at_k": hit_rate,
        "mrr_at_k": mrr,
        "k": topk,
    }


def print_summary(title: str, stats: dict) -> None:
    print(f"\n=== {title} ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:>18}: {v:.6f}")
        else:
            print(f"{k:>18}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Word2Vec Testbench")
    parser.add_argument("--model-path", default="models/word2vec.model", help="Pfad zum Word2Vec-Modell")
    parser.add_argument("--num-pairs", type=int, default=10000, help="Anzahl zufaelliger Produktpaare")
    parser.add_argument("--num-anchors", type=int, default=300, help="Anzahl Zufallsanker fuer Reziprozitaets-Test")
    parser.add_argument("--topk", type=int, default=10, help="Top-K Nachbarn im Reziprozitaets-Test")
    parser.add_argument(
        "--eval-orders-csv",
        default="data/2024-20250001_part_00-001.csv",
        help="CSV mit Order-Events fuer Basket-Retrieval-Test",
    )
    parser.add_argument("--eval-max-orders", type=int, default=10000, help="Maximale Anzahl Baskets fuer Eval")
    parser.add_argument("--eval-k", type=int, default=10, help="K fuer HitRate/MRR Basket-Retrieval-Test")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    args = parser.parse_args()

    t0 = time.time()
    model = Word2Vec.load(args.model_path)
    load_time = time.time() - t0

    print("Word2Vec Testbench")
    print(f"model_path         : {args.model_path}")
    print(f"vocab_size         : {len(model.wv.index_to_key)}")
    print(f"vector_size        : {model.vector_size}")
    print(f"load_time_sec      : {load_time:.3f}")

    _, dist_stats = sample_cosine_distances(
        model=model,
        num_pairs=args.num_pairs,
        seed=args.seed,
    )
    print_summary("Random Pair Cosine Distance", dist_stats)

    recip_stats = reciprocal_neighbor_rate(
        model=model,
        topk=args.topk,
        num_anchors=args.num_anchors,
        seed=args.seed,
    )
    print_summary("Top-K Reciprocity", recip_stats)

    extremes = random_anchor_distance_extremes(
        model=model,
        seed=args.seed,
        topn=10,
    )
    print(f"\n=== Random Anchor Extremes (anchor={extremes['anchor']}) ===")
    print("Top 10 naechste Woerter (niedrigste Cosine-Distanz):")
    for pid, dist in extremes["nearest"]:
        print(f"  {pid:>18}  dist={dist:.6f}")

    print("Top 10 entfernteste Woerter (hoechste Cosine-Distanz):")
    for pid, dist in extremes["farthest"]:
        print(f"  {pid:>18}  dist={dist:.6f}")

    baskets = load_baskets_from_csv(
        csv_path=args.eval_orders_csv,
        max_orders=args.eval_max_orders,
    )
    retrieval_stats = basket_retrieval_metrics(
        model=model,
        baskets=baskets,
        topk=args.eval_k,
    )
    print_summary("Basket Retrieval Quality", retrieval_stats)

    print("\nHinweis:")
    print("- Niedrigere mittlere Cosine-Distanz bedeutet dichtere Embeddings.")
    print("- Hohe Reziprozitaet deutet auf stabile lokale Nachbarschaften hin.")


if __name__ == "__main__":
    main()
