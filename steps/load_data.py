from zenml import step
import polars as pl
import os

def _read_orders_csv_permissive(path: str) -> pl.DataFrame:
    schema = {
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
        schema=schema
    ).filter(
        pl.col("orderid").is_not_null()
        & (pl.col("orderid").str.strip_chars() != "")
        & pl.col("productid").is_not_null()
        & (pl.col("productid").str.strip_chars() != "")
    )

@step
def load_data() -> str:
    source_path = "data/2024-20250001_part_00-001.csv"
    target_path = "data/2024-20250001_part_00-001.parquet"
    products = _read_orders_csv_permissive(source_path)
    products.write_parquet(target_path)
    return target_path

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

@step
def clean_blocked_products(orders_path: str, products_path: str) -> tuple[str, str]:
    products_tmp = products_path + ".tmp"
    orders_tmp = orders_path + ".tmp"

    products = pl.scan_parquet(products_path)
    products = products.filter(pl.col("blocked") == False)
    products.sink_parquet(products_tmp)
    os.replace(products_tmp, products_path)

    valid_product_ids = pl.scan_parquet(products_path).select("productid").unique()

    orders_clean = (
        pl.scan_parquet(orders_path)
        .join(valid_product_ids, on="productid", how="semi")
    )

    orders_clean.sink_parquet(orders_tmp)
    os.replace(orders_tmp, orders_path)

    return orders_path, products_path