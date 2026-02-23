from zenml import step
import polars as pl

def _read_orders_csv_permissive(path: str) -> pl.DataFrame:
    schema_overrides = {
        "orderid": pl.Utf8,
        "productid": pl.Utf8,
        "userid": pl.Utf8,
        "orderdt": pl.Datetime,
        "documentcode": pl.Utf8,
        "documenttype": pl.Utf8,
        "priceperunit": pl.Float64,
        "tax": pl.Float64,
        "currency": pl.Utf8,
        "discountperunit": pl.Float64,
        "origin": pl.Utf8,
        "sellerid": pl.Float64,
        "sellerrouteid": pl.Utf8,
        "discountedpriceperunit": pl.Utf8,
        "quantity": pl.Float64,
        "couponcode": pl.Utf8,
    }

    return pl.read_csv(
        path,
        schema_overrides=schema_overrides
    ).filter(
        pl.col("orderid").is_not_null()
        & (pl.col("orderid").str.strip_chars() != "")
        & pl.col("productid").is_not_null()
        & (pl.col("productid").str.strip_chars() != "")
    )

@step
def load_data_clean() -> str:
    source_path = "data/2024-20250001_part_00-001.csv"
    target_path = "data/2024-20250001_part_00-001.parquet"
    products = _read_orders_csv_permissive(source_path)
    products.write_parquet(target_path)
    return target_path

@step
def load_data() -> str:
    products = pl.read_csv("data/2024-20250001_part_00-001.csv", schema_overrides={
        "priceperunit": pl.Float64,
        "documentcode": pl.Utf8
    })
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


@step
def save_train_test_split(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    train_path: str = "data/train_df.parquet",
    test_path: str = "data/test_df.parquet",
) -> tuple[str, str]:
    train_df.write_parquet(train_path)
    test_df.write_parquet(test_path)
    return train_path, test_path

@step
def save_df(df: pl.DataFrame, path: str) -> str:
    df.write_parquet(path)
    return path