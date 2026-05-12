from zenml import pipeline
from steps.load_data import load_data, load_data_testTrain_seperated, load_products, load_commerces, save_train_test_split, save_df, clean_blocked_products
import steps.train_Word2Vec as word2vec_model
from steps.train_lightGBM import ranker_training_pipeline_fast

@pipeline(enable_cache=False)
def run_pipeline():
    #Pipeline with no split
    data_path_train = load_data()
    commerces_path = load_commerces()
    products_path = load_products()
    data_path_train, products_path = clean_blocked_products(data_path_train, products_path)
    train_df_path = word2vec_model.build_baskets(data_path_train)
    W2Vmodel_path = word2vec_model.train_model(train_df_path)

    LGM_model_path = ranker_training_pipeline_fast(
        orders_path=data_path_train,
        commerces_path=commerces_path,
        products_path=products_path,
        w2v_path=W2Vmodel_path,
        artifacts_dir="artifacts",
        topk=10,
    )



    #Pipeline with seperated split (two .CSV-Files)
    # data_path_train, data_path_test = load_data_testTrain_seperated()
    # commerces_path = load_commerces()
    # products_path = load_products()
    # data_path_train, products_path = clean_blocked_products(data_path_train, products_path)
    # data_path_test, _ = clean_blocked_products(data_path_test, products_path)
    # train_df_path = word2vec_model.build_baskets(data_path_train)
    # W2Vmodel_path = word2vec_model.train_model(train_df_path)

    # LGM_model_path = ranker_training_pipeline_fast(
    #     orders_path=data_path_train,
    #     commerces_path=commerces_path,
    #     products_path=products_path,
    #     w2v_path=W2Vmodel_path,
    #     artifacts_dir="artifacts",
    #     topk=10,
    # )



    # #Pipeline with 80/20 split
    # data_path_train = load_data()
    # commerces_path = load_commerces()
    # products_path = load_products()
    # data_path_train, products_path = clean_blocked_products(data_path_train, products_path)
    # train_df, test_df = word2vec_model.data_split(data_path_train)
    # data_path_train, data_path_test = save_train_test_split(train_df, test_df)
    # train_df_path = word2vec_model.build_baskets(data_path_train)
    # W2Vmodel_path = word2vec_model.train_model(train_df_path)

    # LGM_model_path = ranker_training_pipeline_fast(
    #     orders_path=data_path_train,
    #     commerces_path=commerces_path,
    #     products_path=products_path,
    #     w2v_path=W2Vmodel_path,
    #     artifacts_dir="artifacts",
    #     topk=10,
    # )



    # #Pipeline with monthly split
    # data_path_train = load_data()
    # commerces_path = load_commerces()
    # products_path = load_products()
    # data_path_train, products_path = clean_blocked_products(data_path_train, products_path)
    # train_df, test_df = word2vec_model.data_split_monthly(data_path_train)
    # data_path_train, data_path_test = save_train_test_split(train_df, test_df)
    # train_df_path = word2vec_model.build_baskets(data_path_train)
    # W2Vmodel_path = word2vec_model.train_model(train_df_path)

    # LGM_model_path = ranker_training_pipeline_fast(
    #     orders_path=data_path_train,
    #     commerces_path=commerces_path,
    #     products_path=products_path,
    #     w2v_path=W2Vmodel_path,
    #     artifacts_dir="artifacts",
    #     topk=10,
    # )

if __name__ == "__main__":
    run_pipeline()
