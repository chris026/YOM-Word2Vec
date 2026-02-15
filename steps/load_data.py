from zenml import step
import polars as pl

@step
def load_data() -> str:
    products = pl.read_csv("data/2024-20250001_part_00-001.csv")
    products.write_parquet("data/2024-20250001_part_00-001.parquet")
    return "data/2024-20250001_part_00-001.parquet"

@step
def load_products() -> str:
    products = pl.read_csv("data/products_v2.csv", separator=";")
    products.write_parquet("data/products_v2.parquet")
    return "data/products_v2.parquet"

@step
def load_commerces() -> str:
    commerces = pl.read_csv("data/commerces.csv", separator=";")
    commerces.write_parquet("data/commerces.parquet")
    return "data/commerces.parquet"