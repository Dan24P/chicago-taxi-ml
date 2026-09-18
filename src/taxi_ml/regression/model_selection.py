import numpy as np
import pandas as pd 

import json
import time

from sklearn.pipeline import Pipeline
from taxi_ml.classifaction import prepare_data
from sklearn.model_selection import cross_validate,KFold



from taxi_ml.common.preprocessing import build_preprocessor
from taxi_ml.common.config import feature_set
from models import models
scoring_metrics = ["MAE","r2"]





X_train = pd.read_parquet("data/regression/X_train.parquet")
y_train = pd.read_parquet("data/regression/y_train.parquet")["trip_total"]


validation_metrics = {}
cv = KFold(n_splits=5,shuffle=True,random_state=42)

for set_name,features in feature_set.items():
    preprocessor = build_preprocessor(features["numeric"],features["cat"]  )



    validation_metrics[set_name]= {}


    for name,model in models.items():
        start = time.perf_counter()
        pipeline = Pipeline([ 
        ("preprocessor",preprocessor),
        ("model",model)

        ])
        

        scoring = cross_validate( pipeline,X= X_train, y = y_train ,  cv = cv ,n_jobs = 2,scoring=["neg_mean_absolute_error", "r2"])



        validation_metrics[set_name][name] = {}
        for metric in scoring:
                    mean = scoring[metric].mean()
                    std = scoring[metric].std()
                    validation_metrics[set_name][name][metric] = {
                    "mean": mean,
                    "std": std
                        }
        end = time.perf_counter()
        print(name,set_name,":", end - start, "seconds")
     
        


with open("results/regression/model_selection.json", "w") as f:
    json.dump(validation_metrics, f, indent=2)