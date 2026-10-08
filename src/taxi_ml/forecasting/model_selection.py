import numpy as np
import pandas as pd 

from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import json
from taxi_ml.common.preprocessing import build_preprocessor
from sklearn.pipeline import Pipeline

from sklearn.model_selection import TimeSeriesSplit,cross_validate
import matplotlib.pyplot as plt 
from taxi_ml.forecasting.models import models
from taxi_ml.forecasting.config import feature_set,numerical_features




data_train = pd.read_parquet("data/forecasting/train_data.parquet")
data_validation = pd.read_parquet("data/forecasting/validation_data.parquet")


X_train = data_train[numerical_features ]
y_train = data_train["trip_count"]

X_validation = data_validation[numerical_features]
y_validation = data_validation["trip_count"]



X_validation["lag_1_diff"] = abs(y_validation - X_validation["lag_1"])
X_validation["lag_24_diff"] = abs(y_validation - X_validation["lag_24"])
X_validation["lag_168_diff"] = abs(y_validation - X_validation["lag_168"])
print("MAE 1 hour ", X_validation["lag_1_diff"].mean())
print("MAE 1 day ", X_validation["lag_24_diff"].mean())
print("MAE 1 week ", X_validation["lag_168_diff"].mean())

print(y_validation)
print(X_validation)
model = LinearRegression()
scoring = model.fit(X_train,y_train)
y_pred = model.predict(X_validation[numerical_features])



print("MAE lin reg", mean_absolute_error(y_validation,y_pred))


ridgemodel = Ridge()
scoring = ridgemodel.fit(X_train,y_train)
y_pred = ridgemodel.predict(X_validation[numerical_features])

print("MAE Ridge", mean_absolute_error(y_validation,y_pred))


model  = HistGradientBoostingRegressor()
score = model.fit(X_train,y_train)

y_pred = model.predict(X_validation[numerical_features])

print("MAE HistGradientBoostingRegressor", mean_absolute_error(y_validation,y_pred))


print(X_train)
cv = TimeSeriesSplit(gap = 0, n_splits=5,test_size= 168*4)
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train)):
    train_start = data_train.iloc[train_idx[0]]["timestamp"]
    train_end   = data_train.iloc[train_idx[-1]]["timestamp"]

    val_start = data_train.iloc[val_idx[0]]["timestamp"]
    val_end   = data_train.iloc[val_idx[-1]]["timestamp"]
    print("train start",train_start )
    print("train end",train_end )
    print("val start",val_start )
    print("val end",val_end ,"\n")

scoring_metrics = ["neg_mean_absolute_error", "r2"]
validation_metrics = {}


for set_name,features in feature_set.items():
    validation_metrics[set_name] = {}
    print(set_name)
    features_list = features["numeric"] + features["cat"]
    print(features_list)
    X_train = data_train[features_list]
    X_validation = data_train[features_list]
    for name,model in models.items():
        #model = config["model"]

        preprocessor = build_preprocessor(features["numeric"],features["cat"] )
        pipeline = Pipeline([
            ("preprocessor",preprocessor),
            ("model",model)
        ])


        validation_metrics[set_name][name] = {}
        scoring = cross_validate(pipeline ,X= X_train, y = y_train ,  cv = cv ,n_jobs = 2,scoring=scoring_metrics)
        for metric in scoring_metrics:
            mean = scoring["test_"+metric].mean()
            std = scoring["test_"+metric].std()

            validation_metrics[set_name][name][metric] = {
                                "mean": mean,
                                "std": std
                                    }




for metric in scoring_metrics:   
    fig,ax = plt.subplots()  
    for name,config in models.items():        
        mean = validation_metrics["lags"][name][metric]["mean"]
        std = validation_metrics["lags"][name][metric]["std"]
        ax.errorbar(x = name,y = mean,yerr = std,label = name,fmt = "o")


    ax.set_title(metric)
    fig.savefig("results/forecasting/plots/val_metrics/"+metric+".pdf")


with open("results/forecasting/model_selection.json", "w") as f:
    json.dump(validation_metrics, f, indent=2)