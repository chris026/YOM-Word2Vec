from zenml import pipeline
from steps.load_data import load_data_clean, load_products, load_commerces, save_train_test_split, save_df
import steps.model as word2vec_model
import steps.lightGBM_model as lightGBM_model
from steps.ranker_pipeline_fast import ranker_training_pipeline_fast
from steps.test_model import test_model
import polars as pl

@pipeline(enable_cache=False)
def run_pipeline():
    data_path = load_data_clean()
    commerces_path = load_commerces()
    products_path = load_products()
    baskets_path = word2vec_model.build_baskets(data_path)
    train_df, test_df = word2vec_model.data_split(baskets_path)
    train_df_path, test_df_path = save_train_test_split(train_df, test_df)
    W2Vmodel_path = word2vec_model.train_model(train_df)
    lightGBM_model.prepare_data(W2Vmodel_path, data_path, commerces_path, products_path)
    #steps.model.plot_all_items_2d(model_path)
    #steps.model.test_model(model)


    _, test_df_all_colums = word2vec_model.data_split(data_path)
    test_df_all_colums_path = save_df(test_df_all_colums, "data/test_df_all_colums.parquet")

    LGM_model_path = ranker_training_pipeline_fast(
        orders_path=test_df_all_colums_path,
        commerces_path=commerces_path,
        products_path=products_path,
        w2v_path=W2Vmodel_path,
        artifacts_dir="artifacts",
        topk=20,
    )

    #metrics = test_model(test_df_path, W2Vmodel_path, LGM_model_path)
    #print(metrics)
    

if __name__ == "__main__":
    run_pipeline()
