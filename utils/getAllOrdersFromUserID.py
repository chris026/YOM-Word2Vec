import polars as pl

input_path = "data/train_df_3m.csv"
user_id_filter = "9077130ee9894b2d1e6d3341b341e006"


df = (
    pl
        .scan_csv(input_path)
        .select(["orderid", "productid", "userid", "orderdt"])
        .filter(pl.col("userid") == user_id_filter)
        .sort("orderid")
    )

df.sink_csv("utils/allOrdersFromUser" + user_id_filter + ".csv")