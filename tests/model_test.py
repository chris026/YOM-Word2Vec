from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import polars as pl
from gensim.models import Word2Vec


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Word2Vec testbench without ZenML: evaluates all Word2Vec models in a folder."
    )
    parser.add_argument("--data-path", default=None, help="Path to test parquet with basket column.")
    parser.add_argument("--models-dir", default=None, help="Directory containing Word2Vec model files.")
    parser.add_argument("--ks", default=None, help="Comma-separated cutoff values (for example: 5,10,20).")
    parser.add_argument("--max-baskets", type=int, default=None, help="Optional limit for faster test runs.")
    parser.add_argument("--report-path", default=None, help="Optional markdown report output path.")
    return parser.parse_args()


def parse_ks(raw: str) -> list[int]:
    ks = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value > 0:
            ks.append(value)
    ks = sorted(set(ks))
    if not ks:
        raise ValueError("No valid k values were provided.")
    return ks


def load_baskets(parquet_path: str, max_baskets: int | None = None) -> list[list[str]]:
    df = pl.read_parquet(parquet_path)
    if "basket" not in df.columns:
        raise ValueError(f"Column 'basket' not found in {parquet_path}.")

    if max_baskets is not None and max_baskets > 0:
        df = df.head(max_baskets)

    baskets: list[list[str]] = []
    for row in df.select("basket").iter_rows(named=True):
        raw_basket = row.get("basket") or []
        basket = [str(pid) for pid in raw_basket if pid is not None]
        basket = list(dict.fromkeys(basket))
        if len(basket) >= 2:
            baskets.append(basket)
    return baskets


def discover_word2vec_models(models_dir: str) -> list[Path]:
    root = Path(models_dir)
    if not root.exists():
        raise FileNotFoundError(f"Model directory not found: {models_dir}")

    model_files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".model")
    if not model_files:
        raise FileNotFoundError(f"No '*.model' files found in {models_dir}")
    return model_files


def dcg_at_k(rel: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    score = 0.0
    for i, r in enumerate(rel[:k], start=1):
        score += float(r) / math.log2(i + 1)
    return score


def build_markdown_table(headers: list[str], rows: list[list[str]], right_align_cols: set[int] | None = None) -> str:
    right_align = right_align_cols or set()
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def _format_row(row_vals: list[str]) -> str:
        cells = []
        for i, cell in enumerate(row_vals):
            text = str(cell)
            padded = text.rjust(widths[i]) if i in right_align else text.ljust(widths[i])
            cells.append(f" {padded} ")
        return "|" + "|".join(cells) + "|"

    align_cells = []
    for i, width in enumerate(widths):
        dash_count = max(width, 3)
        if i in right_align:
            align_cells.append(" " + "-" * (dash_count - 1) + ": ")
        else:
            align_cells.append(" :" + "-" * (dash_count - 1) + " ")
    align_row = "|" + "|".join(align_cells) + "|"

    out = [_format_row(headers), align_row]
    out.extend(_format_row(row) for row in rows)
    return "\n".join(out)


def evaluate_word2vec_model(model: Word2Vec, baskets: list[list[str]], ks: list[int]) -> dict:
    wv = model.wv
    max_k = max(ks)

    total_anchors = 0
    oov_anchors = 0
    evaluated_anchors = 0
    no_positive_anchors = 0

    sums: dict[int, dict[str, float]] = {
        k: {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "hitrate": 0.0,
            "mrr": 0.0,
            "map": 0.0,
            "ndcg": 0.0,
            "tp_avg": 0.0,
        }
        for k in ks
    }
    unique_preds: dict[int, set[str]] = {k: set() for k in ks}

    for basket in baskets:
        basket_set = set(basket)
        for anchor in basket:
            total_anchors += 1
            if anchor not in wv:
                oov_anchors += 1
                continue

            positives = {p for p in basket_set if p != anchor and p in wv}
            if not positives:
                no_positive_anchors += 1
                continue

            retrieved = wv.most_similar(anchor, topn=max_k)
            preds = [str(pid) for pid, _ in retrieved if str(pid) != anchor]
            if not preds:
                continue

            evaluated_anchors += 1

            for k in ks:
                topk = preds[:k]
                pred_set = set(topk)
                unique_preds[k].update(topk)

                tp = len(pred_set & positives)
                fp = len(pred_set - positives)
                fn = len(positives - pred_set)

                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
                hitrate = 1.0 if tp > 0 else 0.0

                rr = 0.0
                hits = 0
                ap_sum = 0.0
                rel: list[int] = []
                for rank, pid in enumerate(topk, start=1):
                    is_rel = 1 if pid in positives else 0
                    rel.append(is_rel)
                    if is_rel:
                        hits += 1
                        ap_sum += hits / rank
                        if rr == 0.0:
                            rr = 1.0 / rank

                denom_ap = min(len(positives), k) if positives else 1
                map_k = ap_sum / denom_ap if positives else 0.0
                idcg = dcg_at_k([1] * min(len(positives), k), k)
                ndcg = (dcg_at_k(rel, k) / idcg) if idcg > 0 else 0.0

                sums[k]["precision"] += precision
                sums[k]["recall"] += recall
                sums[k]["f1"] += f1
                sums[k]["hitrate"] += hitrate
                sums[k]["mrr"] += rr
                sums[k]["map"] += map_k
                sums[k]["ndcg"] += ndcg
                sums[k]["tp_avg"] += float(tp)

    denominator = max(evaluated_anchors, 1)
    metrics_at_k = {}
    for k in ks:
        metrics_at_k[k] = {
            "precision": sums[k]["precision"] / denominator,
            "recall": sums[k]["recall"] / denominator,
            "f1": sums[k]["f1"] / denominator,
            "hitrate": sums[k]["hitrate"] / denominator,
            "mrr": sums[k]["mrr"] / denominator,
            "map": sums[k]["map"] / denominator,
            "ndcg": sums[k]["ndcg"] / denominator,
            "avg_true_positives": sums[k]["tp_avg"] / denominator,
            "prediction_coverage": (
                len(unique_preds[k]) / max(len(wv.index_to_key), 1)
            ),
        }

    return {
        "baskets": len(baskets),
        "vocab_size": len(wv.index_to_key),
        "vector_size": model.vector_size,
        "total_anchors": total_anchors,
        "oov_anchors": oov_anchors,
        "oov_anchor_rate": (oov_anchors / total_anchors) if total_anchors else 0.0,
        "anchors_without_positive": no_positive_anchors,
        "evaluated_anchors": evaluated_anchors,
        "metrics_at_k": metrics_at_k,
    }


def format_model_section(name: str, result: dict, ks: list[int]) -> str:
    lines = []
    lines.append(f"## {name}")
    lines.append(f"- baskets: {result['baskets']}")
    lines.append(f"- vocab_size: {result['vocab_size']}")
    lines.append(f"- vector_size: {result['vector_size']}")
    lines.append(f"- total_anchors: {result['total_anchors']}")
    lines.append(f"- evaluated_anchors: {result['evaluated_anchors']}")
    lines.append(f"- oov_anchors: {result['oov_anchors']}")
    lines.append(f"- oov_anchor_rate: {result['oov_anchor_rate']:.4f}")
    lines.append("")

    keys = [
        "precision",
        "recall",
        "f1",
        "hitrate",
        "mrr",
        "map",
        "ndcg",
        "avg_true_positives",
        "prediction_coverage",
    ]
    table_headers = ["metric"] + [f"@{k}" for k in ks]
    table_rows: list[list[str]] = []
    for metric_name in keys:
        row = [metric_name]
        for k in ks:
            value = result["metrics_at_k"][k][metric_name]
            row.append(f"{value:.4f}")
        table_rows.append(row)
    lines.append(
        build_markdown_table(
            headers=table_headers,
            rows=table_rows,
            right_align_cols={i for i in range(1, len(table_headers))},
        )
    )
    lines.append("")
    return "\n".join(lines)


def build_comparison_table(results: dict[str, dict], ranking_k: int) -> str:
    lines = []
    lines.append(f"## Model Comparison (sorted by NDCG@{ranking_k})")

    ordered = sorted(
        results.items(),
        key=lambda x: x[1]["metrics_at_k"][ranking_k]["ndcg"],
        reverse=True,
    )

    table_headers = ["model", "ndcg", "hitrate", "precision", "recall", "mrr", "map", "eval_anchors"]
    table_rows: list[list[str]] = []
    for model_name, res in ordered:
        m = res["metrics_at_k"][ranking_k]
        table_rows.append(
            [
                model_name,
                f"{m['ndcg']:.4f}",
                f"{m['hitrate']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['mrr']:.4f}",
                f"{m['map']:.4f}",
                str(res["evaluated_anchors"]),
            ]
        )
    lines.append(
        build_markdown_table(
            headers=table_headers,
            rows=table_rows,
            right_align_cols={1, 2, 3, 4, 5, 6, 7},
        )
    )
    lines.append("")
    return "\n".join(lines)


def print_console_summary(results: dict[str, dict], ranking_k: int) -> None:
    print("\n=== WORD2VEC TESTBENCH REPORT ===")
    print(f"Ranking metric for comparison: NDCG@{ranking_k}")
    ordered = sorted(
        results.items(),
        key=lambda x: x[1]["metrics_at_k"][ranking_k]["ndcg"],
        reverse=True,
    )
    for idx, (model_name, res) in enumerate(ordered, start=1):
        m = res["metrics_at_k"][ranking_k]
        print(
            f"{idx:>2}. {model_name} | "
            f"NDCG@{ranking_k}={m['ndcg']:.4f} | "
            f"HitRate@{ranking_k}={m['hitrate']:.4f} | "
            f"Precision@{ranking_k}={m['precision']:.4f} | "
            f"Recall@{ranking_k}={m['recall']:.4f} | "
            f"MRR@{ranking_k}={m['mrr']:.4f} | "
            f"EvalAnchors={res['evaluated_anchors']}"
        )


def main() -> None:
    args = parse_args()
    root = project_root()

    data_path = Path(args.data_path) if args.data_path else (root / "data" / "test_df.parquet")
    models_dir = Path(args.models_dir) if args.models_dir else (root / "models")
    report_path = Path(args.report_path) if args.report_path else (root / "tests" / "word2vec_report.md")
    ks = parse_ks(args.ks if args.ks else "5,10,20")
    ranking_k = 10 if 10 in ks else ks[0]

    baskets = load_baskets(str(data_path), args.max_baskets)
    model_files = discover_word2vec_models(str(models_dir))

    print(f"Loaded baskets: {len(baskets)}")
    print(f"Discovered model files: {len(model_files)}")

    results: dict[str, dict] = {}
    load_failures: list[tuple[str, str]] = []

    for model_path in model_files:
        model_name = model_path.name
        print(f"\nEvaluating {model_name} ...")
        t0 = time.time()
        try:
            model = Word2Vec.load(str(model_path))
        except Exception as exc:
            load_failures.append((model_name, str(exc)))
            print(f"  -> skipped (load error: {exc})")
            continue

        result = evaluate_word2vec_model(model=model, baskets=baskets, ks=ks)
        result["eval_time_sec"] = time.time() - t0
        results[model_name] = result
        print(
            f"  -> done in {result['eval_time_sec']:.2f}s "
            f"(eval_anchors={result['evaluated_anchors']}, "
            f"ndcg@{ranking_k}={result['metrics_at_k'][ranking_k]['ndcg']:.4f})"
        )

    if not results:
        raise RuntimeError("No Word2Vec models could be evaluated successfully.")

    print_console_summary(results=results, ranking_k=ranking_k)

    report_lines = []
    report_lines.append("# Word2Vec Multi-Model Test Report")
    report_lines.append("")
    report_lines.append(f"- data_path: `{data_path}`")
    report_lines.append(f"- models_dir: `{models_dir}`")
    report_lines.append(f"- baskets_used: {len(baskets)}")
    report_lines.append(f"- ks: {ks}")
    report_lines.append("")
    report_lines.append(build_comparison_table(results, ranking_k=ranking_k))
    for model_name in sorted(results.keys()):
        report_lines.append(format_model_section(model_name, results[model_name], ks))

    if load_failures:
        report_lines.append("## Skipped Files")
        for file_name, reason in load_failures:
            report_lines.append(f"- {file_name}: {reason}")
        report_lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nMarkdown report written to: {report_path}")
    if load_failures:
        print("\nSkipped files:")
        for file_name, reason in load_failures:
            print(f"- {file_name}: {reason}")


if __name__ == "__main__":
    main()
