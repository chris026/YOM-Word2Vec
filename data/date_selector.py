from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_CSV = Path(__file__).resolve().parent / "2024-20250001_part_00-001.csv"
TEST_OUTPUT_CSV = Path(__file__).resolve().parent / "test_df.csv"
TRAIN_OUTPUT_CSV = Path(__file__).resolve().parent / "train_df.csv"

# Anzahl Wochen im Test-Split (vom neuesten Datum rueckwaerts)
TEST_WEEKS = 4

# Anzahl Wochen im Train-Split direkt vor dem Test-Split
TRAIN_WEEKS = 12

# Groesse der CSV-Chunks fuer speichereffizientes Processing
CHUNK_SIZE = 500_000

ORDER_DATE_COLUMN = "orderdt"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
READ_DTYPES = {"documentcode": "string"}


def find_latest_order_date(csv_path: Path) -> pd.Timestamp:
    latest_date: pd.Timestamp | None = None

    for chunk in pd.read_csv(
        csv_path,
        usecols=[ORDER_DATE_COLUMN],
        dtype={ORDER_DATE_COLUMN: "string"},
        chunksize=CHUNK_SIZE,
    ):
        parsed_dates = pd.to_datetime(
            chunk[ORDER_DATE_COLUMN],
            format=DATE_FORMAT,
            errors="coerce",
        )
        chunk_max = parsed_dates.max()

        if pd.notna(chunk_max) and (latest_date is None or chunk_max > latest_date):
            latest_date = chunk_max

    if latest_date is None:
        raise ValueError(
            f"Kein gueltiges Datum in Spalte '{ORDER_DATE_COLUMN}' gefunden."
        )

    return latest_date


def build_splits() -> None:
    if TEST_WEEKS <= 0:
        raise ValueError("TEST_WEEKS muss groesser als 0 sein.")
    if TRAIN_WEEKS <= 0:
        raise ValueError("TRAIN_WEEKS muss groesser als 0 sein.")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {INPUT_CSV}")

    latest_order_date = find_latest_order_date(INPUT_CSV)

    test_start_exclusive = latest_order_date - pd.Timedelta(weeks=TEST_WEEKS)
    train_end_inclusive = test_start_exclusive
    train_start_exclusive = train_end_inclusive - pd.Timedelta(weeks=TRAIN_WEEKS)

    for output_path in (TEST_OUTPUT_CSV, TRAIN_OUTPUT_CSV):
        if output_path.exists():
            output_path.unlink()

    header_columns = pd.read_csv(INPUT_CSV, nrows=0).columns
    write_test_header = True
    write_train_header = True
    test_rows = 0
    train_rows = 0

    for chunk in pd.read_csv(
        INPUT_CSV,
        chunksize=CHUNK_SIZE,
        dtype=READ_DTYPES,
    ):
        parsed_dates = pd.to_datetime(
            chunk[ORDER_DATE_COLUMN],
            format=DATE_FORMAT,
            errors="coerce",
        )

        test_mask = parsed_dates.gt(test_start_exclusive) & parsed_dates.le(
            latest_order_date
        )
        train_mask = parsed_dates.gt(train_start_exclusive) & parsed_dates.le(
            train_end_inclusive
        )

        if test_mask.any():
            test_chunk = chunk.loc[test_mask]
            test_chunk.to_csv(
                TEST_OUTPUT_CSV,
                mode="a",
                header=write_test_header,
                index=False,
            )
            write_test_header = False
            test_rows += len(test_chunk)

        if train_mask.any():
            train_chunk = chunk.loc[train_mask]
            train_chunk.to_csv(
                TRAIN_OUTPUT_CSV,
                mode="a",
                header=write_train_header,
                index=False,
            )
            write_train_header = False
            train_rows += len(train_chunk)

    if test_rows == 0:
        pd.DataFrame(columns=header_columns).to_csv(TEST_OUTPUT_CSV, index=False)
    if train_rows == 0:
        pd.DataFrame(columns=header_columns).to_csv(TRAIN_OUTPUT_CSV, index=False)

    print(f"Neuestes orderdt: {latest_order_date}")
    print(
        f"test_df.csv: {test_rows} Zeilen fuer {TEST_WEEKS} Wochen "
        f"({test_start_exclusive} .. {latest_order_date})"
    )
    print(
        f"train_df.csv: {train_rows} Zeilen fuer {TRAIN_WEEKS} Wochen "
        f"({train_start_exclusive} .. {train_end_inclusive})"
    )


if __name__ == "__main__":
    build_splits()
