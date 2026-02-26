import argparse
from pathlib import Path

import polars as pl


def _get_row_count(path: Path, lazy_df: pl.LazyFrame) -> int:
    """Liest die Zeilenanzahl effizient aus Parquet-Metadaten; faellt sonst auf Count zurueck."""
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        # Fallback, falls pyarrow nicht verfuegbar ist oder Metadatenzugriff fehlschlaegt.
        return int(lazy_df.select(pl.len()).collect(engine="streaming").item())


def inspect_parquet(file_path: str, n: int = 5) -> None:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    # Lazy scan: liest zunaechst nur Metadaten und laedt Daten erst bei collect().
    lazy_df = pl.scan_parquet(str(path))

    print(f"Datei: {path}")
    print(f"Zeilen: {_get_row_count(path, lazy_df)}")

    print("\nSchema:")
    schema = lazy_df.collect_schema()
    for name, dtype in schema.items():
        print(f"- {name}: {dtype}")

    print(f"\nHead({n}):")
    head_df = lazy_df.limit(n).collect(engine="streaming")
    print(head_df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zeigt Schema und die ersten Zeilen einer Parquet-Datei an."
    )
    parser.add_argument(
        "--file",
        help="Pfad zur .parquet-Datei",
        default="artifacts/train.parquet")
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Anzahl der Zeilen fuer die Ausgabe (Default: 5)",
    )
    args = parser.parse_args()

    inspect_parquet(args.file, args.rows)


if __name__ == "__main__":
    main()
