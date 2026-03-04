from zenml import pipeline
from steps.load_data import load_data, load_data_testTrain_seperated, load_products, load_commerces, save_train_test_split, save_df, clean_blocked_products
import steps.train_Word2Vec as word2vec_model
from steps.train_lightGBM import ranker_training_pipeline_fast
from steps.test_model import test_model

@pipeline(enable_cache=False)
def run_pipeline():
    data_path_train, data_path_test = load_data_testTrain_seperated()
    commerces_path = load_commerces()
    products_path = load_products()
    data_path_train, products_path = clean_blocked_products(data_path_train, products_path)
    data_path_test, _ = clean_blocked_products(data_path_test, products_path)
    train_df_path = word2vec_model.build_baskets(data_path_train)
    #train_df, test_df = word2vec_model.data_split(baskets_path)
    #train_df_path, test_df_path = save_train_test_split(train_df, test_df)
    W2Vmodel_path = word2vec_model.train_model(train_df_path)
    #steps.model.plot_all_items_2d(model_path)
    #steps.model.test_model(model)


    #train_df_all_colums, _ = word2vec_model.data_split(data_path_train)
    #train_df_all_colums_path = save_df(train_df_all_colums, "data/train_df_all_colums.parquet")

    data_path_train_LGM, _ = load_data_testTrain_seperated()

    LGM_model_path = ranker_training_pipeline_fast(
        orders_path=data_path_train_LGM,
        commerces_path=commerces_path,
        products_path=products_path,
        w2v_path=W2Vmodel_path,
        artifacts_dir="artifacts",
        topk=10,
    )

    #metrics = test_model(test_df_path, W2Vmodel_path, LGM_model_path)
    #print(metrics)
    

if __name__ == "__main__":
    run_pipeline()
