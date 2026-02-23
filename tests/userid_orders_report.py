from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

TARGET_USER_ID = "5d31a286f31582e9d31135a0dccba03e"
OUTPUT_MD_PATH = "tests/user_orders_report.md"
DEFAULT_PARQUET_CANDIDATES = [
    "data/20242024-20250001_part_00-001.parquet",
    "data/2024-20250001_part_00-001.parquet",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filtert Bestellungen fuer eine userid und schreibt orderid + productid als Markdown-Report."
        )
    )
    parser.add_argument("--userid", default=TARGET_USER_ID, help="Zu filternde userid")
    parser.add_argument(
        "--parquet-path",
        default=None,
        help="Optionaler Parquet-Pfad. Wenn leer, werden Standardpfade probiert.",
    )
    parser.add_argument("--output-md", default=OUTPUT_MD_PATH, help="Ausgabe-Report (.md)")
    return parser.parse_args()


def resolve_parquet_path(cli_path: str | None) -> Path:
    if cli_path:
        path = Path(cli_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet-Datei nicht gefunden: {path}")
        return path

    for candidate in DEFAULT_PARQUET_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path

    candidates = ", ".join(DEFAULT_PARQUET_CANDIDATES)
    raise FileNotFoundError(
        "Keine Standard-Parquet-Datei gefunden. Gepruefte Pfade: " + candidates
    )


def read_rows(path: Path) -> list[tuple[str, str, str]]:
    cols = ["userid", "orderid", "productid"]

    try:
        import polars as pl

        df = pl.read_parquet(str(path), columns=cols)
        return [
            (
                "" if r[0] is None else str(r[0]),
                "" if r[1] is None else str(r[1]),
                "" if r[2] is None else str(r[2]),
            )
            for r in df.iter_rows()
        ]
    except ModuleNotFoundError:
        pass

    try:
        import pandas as pd

        df = pd.read_parquet(path, columns=cols)
        rows: list[tuple[str, str, str]] = []
        for user_id, order_id, product_id in df.itertuples(index=False, name=None):
            rows.append(
                (
                    "" if user_id is None else str(user_id),
                    "" if order_id is None else str(order_id),
                    "" if product_id is None else str(product_id),
                )
            )
        return rows
    except ImportError as exc:
        raise RuntimeError(
            "Kein Parquet-Reader verfuegbar. Installiere z.B. 'polars' oder 'pyarrow'."
        ) from exc


def group_products_by_order(rows: list[tuple[str, str, str]], user_id: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    seen_per_order: dict[str, set[str]] = defaultdict(set)

    for row_user, order_id, product_id in rows:
        if row_user != user_id:
            continue
        if not order_id or not product_id:
            continue
        if product_id in seen_per_order[order_id]:
            continue

        grouped[order_id].append(product_id)
        seen_per_order[order_id].add(product_id)

    return dict(grouped)


def build_markdown(user_id: str, parquet_path: Path, grouped: dict[str, list[str]]) -> str:
    total_products = sum(len(products) for products in grouped.values())
    timestamp = datetime.now().isoformat(timespec="seconds")

    lines: list[str] = [
        f"# Bestellungen fuer userid `{user_id}`",
        "",
        f"- Quelle: `{parquet_path.as_posix()}`",
        f"- Generiert am: `{timestamp}`",
        f"- Anzahl orderid: `{len(grouped)}`",
        f"- Anzahl productid (unique pro order): `{total_products}`",
        "",
    ]

    if not grouped:
        lines.append(f"Keine Bestellungen fuer userid `{user_id}` gefunden.")
        lines.append("")
        return "\n".join(lines)

    for order_id in sorted(grouped):
        lines.append(f"## orderid `{order_id}`")
        lines.append("")
        lines.append("| Nr | productid |")
        lines.append("|---:|---|")
        for idx, product_id in enumerate(grouped[order_id], start=1):
            lines.append(f"| {idx} | `{product_id}` |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    user_id = str(args.userid)
    parquet_path = resolve_parquet_path(args.parquet_path)
    rows = read_rows(parquet_path)
    grouped = group_products_by_order(rows, user_id)

    markdown = build_markdown(user_id, parquet_path, grouped)
    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Report geschrieben: {output_path}")
    print(f"orderid gefunden: {len(grouped)}")


if __name__ == "__main__":
    main()
