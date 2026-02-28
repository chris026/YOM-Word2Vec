from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_CSV = Path(__file__).resolve().parent / "2024-20250001_part_00-001.csv"
TEST_OUTPUT_CSV = Path(__file__).resolve().parent / "test_df_1m.csv"
TRAIN_OUTPUT_CSV = Path(__file__).resolve().parent / "train_df_3m.csv"

# Anzahl Kalendermonate im Test-Split (inkl. aktuellem, ggf. unvollstaendigem Monat)
TEST_MONTHS = 1

# Anzahl Kalendermonate im Train-Split direkt vor dem Test-Split
TRAIN_MONTHS = 3

# Groesse der CSV-Chunks fuer speichereffizientes Processing
CHUNK_SIZE = 1_000_000

ORDER_DATE_COLUMN = "orderdt"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
READ_DTYPES = {"documentcode": "string"}


def find_latest_order_date() -> pd.Timestamp:
    return pd.Timestamp("2025-11-10 00:00:00")


def build_splits() -> None:
    if TEST_MONTHS <= 0:
        raise ValueError("TEST_MONTHS muss groesser als 0 sein.")
    if TRAIN_MONTHS <= 0:
        raise ValueError("TRAIN_MONTHS muss groesser als 0 sein.")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {INPUT_CSV}")

    latest_order_date = find_latest_order_date()

    latest_month_start = latest_order_date.to_period("M").to_timestamp()
    test_start_inclusive = latest_month_start - pd.DateOffset(months=TEST_MONTHS - 1)
    test_end_inclusive = latest_order_date
    train_end_exclusive = test_start_inclusive
    train_start_inclusive = train_end_exclusive - pd.DateOffset(months=TRAIN_MONTHS)

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
            errors="raise",
        )

        test_mask = parsed_dates.ge(test_start_inclusive) & parsed_dates.le(
            test_end_inclusive
        )
        train_mask = parsed_dates.ge(train_start_inclusive) & parsed_dates.lt(
            train_end_exclusive
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
        f"test_df.csv: {test_rows} Zeilen fuer {TEST_MONTHS} Monate "
        f"({test_start_inclusive} .. {test_end_inclusive})"
    )
    print(
        f"train_df.csv: {train_rows} Zeilen fuer {TRAIN_MONTHS} Monate "
        f"({train_start_inclusive} .. {train_end_exclusive}, Ende exklusiv)"
    )


if __name__ == "__main__":
    build_splits()
