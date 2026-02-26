from pathlib import Path

import polars as pl

PARQUET_PATH = "artifacts/train.parquet"
CSV_PATH = "artifacts/parquet_to.csv"


def prepare_for_csv(df: pl.DataFrame) -> pl.DataFrame:
    transforms = []

    for name, dtype in df.schema.items():
        if isinstance(dtype, pl.List):
            transforms.append(
                pl.col(name)
                .list.eval(pl.element().cast(pl.String))
                .list.join("|")
                .alias(name)
            )
        elif isinstance(dtype, pl.Struct):
            transforms.append(pl.col(name).struct.json_encode().alias(name))

    if transforms:
        return df.with_columns(transforms)
    return df


def main() -> None:
    parquet_path = Path(PARQUET_PATH)
    csv_path = Path(CSV_PATH)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet-Datei nicht gefunden: {parquet_path}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(parquet_path)
    df = prepare_for_csv(df)
    df.write_csv(csv_path)

    print(f"CSV geschrieben: {csv_path}")


if __name__ == "__main__":
    main()
