from zenml import step
import polars as pl

@step(enable_cache=False)
def prepare_data(W2Vmodel_path: str, data_path, commerces_path, products_path):
    orders = pl.scan_parquet(data_path)
    commerces = pl.scan_parquet(commerces_path)
    #commerces_cols_needed = ["commune","channel"]
    #commerces = commerces.select(commerces_cols_needed).drop_nulls()
    products = pl.scan_parquet(products_path)

    # global popularity
    pop_global = orders.group_by("productid").agg(pl.count().alias("pop_global"))
    pop_store = orders.group_by(["userid","productid"]).agg(pl.count().alias("pop_store"))
    pop_region = orders.group_by(["region","productid"]).agg(pl.count().alias("pop_region"))
    pop_subch = orders.group_by(["subchannel","productid"]).agg(pl.count().alias("pop_subch"))
    pop_origin = orders.group_by(["origin","productid"]).agg(pl.count().alias("pop_origin"))

    print(pop_global.head(5).collect())
    print(pop_global.filter(pl.col("productid") == "000051-007").collect())
    