import duckdb
import numpy as np
import pandas as pd 

import json
import time

from sklearn.pipeline import Pipeline
from taxi_ml.classifaction import prepare_data
from sklearn.model_selection import cross_validate,KFold





from taxi_ml.common.preprocessing import build_preprocessor



con = duckdb.connect("data/taxi.db")



numerical_features = ["trip_seconds","trip_miles","hour_sin","hour_cos"]
categorical_features = ["weekday","company"]
#
#
feature_set = {}
feature_set["base"] = { "numeric" : numerical_features , "cat" : [] } 
feature_set["base_weekday"] = { "numeric" : numerical_features , "cat" : ["weekday"] } 
feature_set["base_company"] = { "numeric" : numerical_features , "cat" : ["company"] } 
feature_set["base_weekday_company"] = {
    "numeric": numerical_features,
    "cat": ["weekday", "company"]
}


X_train = pd.read_parquet("data/X_train.parquet")
X_test = pd.read_parquet("data/X_test.parquet")

y_train = pd.read_parquet("data/y_train.parquet")["trip_total"]
y_test = pd.read_parquet("data/y_test.parquet")["trip_total"]

validation_metrics = {}
cv = KFold(n_splits=5,shuffle=True,random_state=42)

for set_name,features in feature_set.items():
    preprocessor = build_preprocessor(features["numeric"],features["cat"]  )



    validation_metrics[set_name]= {}


    for name,model in prepare_data.items():
        start = time.perf_counter()
        pipeline = Pipeline([ 
        ("preprocessor",preprocessor),
        ("model",model)

        ])
        

        scoring = cross_validate( pipeline,X= X_train, y = y_train ,  cv = cv ,n_jobs = 2,scoring=["neg_mean_absolute_error", "r2"])

        mean_abs_err = -scoring["test_neg_mean_absolute_error"]

        test_r2 = scoring["test_r2"]

        validation_metrics[set_name][name] = {
            "MAE": mean_abs_err.mean(),
            "MAE_std" :mean_abs_err.std(),
            "r2": test_r2.mean(),
            "r2_std":test_r2.std()
        }
        end = time.perf_counter()
        print(name,set_name,":", end - start, "seconds")
     
        


with open("results/validation_metrics.json", "w") as f:
    json.dump(validation_metrics, f, indent=2)