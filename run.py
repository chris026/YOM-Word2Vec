from zenml import pipeline
from steps.load_data import load_data, load_data_testTrain_seperated, load_products, load_commerces, save_train_test_split, clean_blocked_products
import steps.train_Word2Vec as word2vec_model
from steps.train_lightGBM import ranker_training_pipeline_fast

# Set to True when the order data spans several months.
# False → load_data() + data_split()          (random 80 / 20 split)
# True  → load_data_testTrain_seperated()     (pre-split train/test CSVs)
#        + data_split_monthly()               (last 2 calendar months held out)
MULTI_MONTH_MODE = False


@pipeline(enable_cache=False)
def run_pipeline():
    """Run the full training pipeline end-to-end.

    The pipeline has two modes controlled by ``MULTI_MONTH_MODE`` at the top
    of this file:

    **MULTI_MONTH_MODE = False** (default)
      1. Load all orders from a single CSV (``load_data``).
      2. Remove blocked products and their orders.
      3. Build baskets and split 80 / 20 at random (``data_split``).
      4. Train Word2Vec on the training baskets.
      5. Train the LightGBM ranker.

    **MULTI_MONTH_MODE = True**
      1. Load pre-split train and test CSVs (``load_data_testTrain_seperated``).
      2. Remove blocked products from both train and test data.
      3. Build baskets and split by calendar month (``data_split_monthly``),
         holding out the last 2 months as the test set.
      4. Train Word2Vec on the training baskets.
      5. Train the LightGBM ranker on the pre-split training data.

    Trained models are saved to ``models/word2vec.model`` and
    ``models/lgbm_ranker.txt``.
    """
    commerces_path = load_commerces()
    products_path = load_products()

    if MULTI_MONTH_MODE:
        data_path_train, data_path_test = load_data_testTrain_seperated()
        data_path_train, products_path = clean_blocked_products(data_path_train, products_path)
        data_path_test, _ = clean_blocked_products(data_path_test, products_path)
        train_df_path = word2vec_model.build_baskets(data_path_train)
        train_df, test_df = word2vec_model.data_split_monthly(train_df_path)
        data_path_train_LGM, _ = load_data_testTrain_seperated()
    else:
        data_path_train = load_data()
        data_path_train, products_path = clean_blocked_products(data_path_train, products_path)
        train_df_path = word2vec_model.build_baskets(data_path_train)
        train_df, test_df = word2vec_model.data_split(train_df_path)
        data_path_train_LGM = load_data()

    train_df_path, test_df_path = save_train_test_split(train_df, test_df)
    W2Vmodel_path = word2vec_model.train_model(train_df_path)

    LGM_model_path = ranker_training_pipeline_fast(
        orders_path=data_path_train_LGM,
        commerces_path=commerces_path,
        products_path=products_path,
        w2v_path=W2Vmodel_path,
        artifacts_dir="artifacts",
        topk=10,
    )


if __name__ == "__main__":
    run_pipeline()
