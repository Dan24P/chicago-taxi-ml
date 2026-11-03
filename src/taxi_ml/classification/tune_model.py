import pandas as pd 

import json

import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate,StratifiedKFold,RandomizedSearchCV
from datetime import datetime




from taxi_ml.common.preprocessing import build_preprocessor
from taxi_ml.common.config import feature_set
from taxi_ml.classification.models import models
from pathlib import Path


def main():

    Path("models/classification").mkdir(parents=True, exist_ok=True)
    Path("results/classification/Tuning").mkdir(
    parents=True, exist_ok=True
    )
    Path("results/classification/plots").mkdir(
    parents=True, exist_ok=True
    )

    param_grid = {
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "model__max_iter": [30,50,80,100, 150],
        "model__max_leaf_nodes": [ 31, 63,95,127],
        "model__min_samples_leaf": [5,10, 15,20],
        "model__l2_regularization": [0.0, 0.1, 0.2,0.5],
        "model__max_features": [0.25,0.5, 0.75],
    }
    scoring_metrics = [ "accuracy","precision","recall","f1","roc_auc","average_precision"]


    X_train = pd.read_parquet("data/classification/X_train.parquet")
    y_train = pd.read_parquet("data/classification/y_train.parquet")["high_fare"]

    metrics = {}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    set_name = "base_company"
    features = feature_set[set_name]
    preprocessor = build_preprocessor(features["numeric"],features["cat"]  )

    model = models["gradient_boosting"]["model"]
    name = "gradient_boosting"

    #for name,model in [("HistGradientBoostingClassifier")]:

    pipeline = Pipeline([ 
            ("preprocessor",preprocessor),
            ("model",model)

            ])

    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=30,scoring="average_precision",cv = cv , random_state=42,n_jobs=1)



    search.fit(X_train, y_train)
    joblib.dump(search.best_estimator_,"models/classification/"+name+"_"+set_name+"_bestEst.joblib")


    bestParams = search.best_params_
    bestScore = search.best_score_
    bestEstimator = search.best_estimator_
    bestIndex = int(search.best_index_)





    tuning_results = {
                "train_length": len(X_train),
                "feature_set": set_name,
                "id": timestamp,
                "CV_Folds" : cv.get_n_splits(),
                "best_index": bestIndex,
        "best_params": bestParams,
        "best_CV_avg_prec": bestScore,
        "best_CV_avg_prec_std": search.cv_results_["std_test_score"][bestIndex]
    }
    metrics[name] = tuning_results




    with open("results/classification/Tuning/"+timestamp+".json", "w") as f:
        json.dump(metrics, f, indent=2)
if __name__=="__main__":
    main()