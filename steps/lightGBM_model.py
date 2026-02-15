from zenml import step
import polars as pl

@step
def prepare_data(W2Vmodel_path: str, data_path, commerces_path, products_path):
    orders = pl.scan_parquet(data_path)
    commerces = pl.scan_parquet(commerces_path)
    #commerces_cols_needed = ["commune","channel"]
    #commerces = commerces.select(commerces_cols_needed).drop_nulls()
    products = pl.scan_parquet(products_path)

    # global popularity
    pop_global = orders.group_by("productid")
    pop_store = orders.group_by(["userid","productid"])
    pop_region = orders.group_by(["region","productid"])
    pop_subch = orders.group_by(["subchannel","productid"])
    pop_origin = orders.group_by(["origin","productid"])

    print(pop_global.head(5).collect())