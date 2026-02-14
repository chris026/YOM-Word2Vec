from zenml import step
import polars as pl

@step
def load_data() -> pl.DataFrame:
    return pl.read_csv("data/2024-20250001_part_00-001.csv")

@step
def safe_as_parqet(data: pl.DataFrame):
    data.write_parquet("data/data.parquet")