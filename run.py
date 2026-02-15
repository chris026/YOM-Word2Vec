from zenml import pipeline
from steps.load_data import load_data, load_products, load_commerces
import steps.model as word2vec_model
import steps.lightGBM_model as lightGBM_model

@pipeline(enable_cache=True)
def run_pipeline():
    data_path = load_data()
    products_path = load_products()
    commerces_path = load_commerces()
    baskets = word2vec_model.build_baskets(data_path)
    train_df, test_df = word2vec_model.data_split(baskets)
    W2Vmodel_path = word2vec_model.train_model(train_df)
    lightGBM_model.prepare_data(W2Vmodel_path, data_path, commerces_path, products_path)
    #steps.model.plot_all_items_2d(model_path)
    #steps.model.test_model(model)
    

if __name__ == "__main__":
    run_pipeline()