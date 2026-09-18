
import pandas as pd 
import matplotlib.pyplot as plt 
import json
import time

from sklearn.pipeline import Pipeline
from taxi_ml.classifaction import prepare_data
from sklearn.model_selection import cross_validate,StratifiedKFold





from taxi_ml.common.preprocessing import build_preprocessor
from taxi_ml.common.config import feature_set
from models import models



scoring_metrics = [ "accuracy","precision","recall","f1","roc_auc","average_precision"]


X_train = pd.read_parquet("data/classification/X_train.parquet")
y_train = pd.read_parquet("data/classification/y_train.parquet")["high_fare"]




validation_metrics = {}
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

for set_name,features in feature_set.items():
    


    validation_metrics[set_name]= {}


    for name,config in models.items():

        model = config["model"]
        scale_numeric = config["scale"]
        start = time.perf_counter()

        preprocessor = build_preprocessor(features["numeric"],features["cat"],  config["scale"] )

        pipeline = Pipeline([ 
        ("preprocessor",preprocessor),
        ("model",model)

        ])

        scoring = cross_validate( pipeline,X= X_train, y = y_train ,  cv = cv ,n_jobs = 2,scoring=scoring_metrics)


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

fig,ax = plt.subplots()

for metric in scoring:     
    for name,config in models.items():        
        mean = validation_metrics["base_weekday_company"][name][metric]["mean"]
        std = validation_metrics["base_weekday_company"][name][metric]["std"]
        ax.errorbar(mean,yerr = std,label = name)


    fig.set_title(metric)
    fig.savefile("results/classification/plots/"+metric+".pdf")
    fig.close()


with open("results/classification/model_selection.json", "w") as f:
    json.dump(validation_metrics, f, indent=2)