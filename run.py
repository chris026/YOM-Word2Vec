from zenml import pipeline
from steps.load_data import load_data, safe_as_parqet
import steps.model

@pipeline(enable_cache=False)
def run_pipeline():
    df = load_data()
    safe_as_parqet(df)
    baskets = steps.model.build_baskets()
    train_df, test_df = steps.model.data_split(baskets)
    model = steps.model.train_model(train_df)
    #steps.model.plot_all_items_2d(model)
    #steps.model.test_model(model)

if __name__ == "__main__":
    run_pipeline()