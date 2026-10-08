import joblib
import pandas as pd 
from  taxi_ml.forecasting.config import feature_set


def main():

    set_name = "lags_hour_weekday"

    model = joblib.load("models/forecasting/gradient_boosting_lags_hour_weekday_bestEst.joblib")

    data_test = pd.read_parquet("data/forecasting/test_data.parquet")


    feature_list = feature_set[set_name]["numeric"]  + feature_set[set_name]["cat"] 
    X_test = data_test[feature_list]

    y_pred = model.predict(X_test)

    print(data_test[["timestamp", "trip_count"]].head(10))

if __name__ == "__main__":
    main()