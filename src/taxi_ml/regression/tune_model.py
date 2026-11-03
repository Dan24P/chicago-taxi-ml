from pathlib import Path


import joblib
import json
import pandas as pd


from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold

from sklearn.model_selection import RandomizedSearchCV

from datetime import datetime

from taxi_ml.common.preprocessing import build_preprocessor
from taxi_ml.common.config import feature_set
from taxi_ml.regression.models import models


def main():
    Path('models/regression').mkdir(parents=True, exist_ok=True)
    Path('results/regression/Tuning').mkdir(parents=True, exist_ok=True)

    param_grid = {
        "model__n_estimators": [20,30,40,50,100],
        "model__max_depth": [5,8,10,20],
        "model__min_samples_leaf": [1,2,5,10],
        "model__max_features": [0.75,1.0,"sqrt"]
    }






    X_train = pd.read_parquet("data/regression/X_train.parquet")
    y_train = pd.read_parquet("data/regression/y_train.parquet")["trip_total"]


    metrics = {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    set_name = "base_weekday_company"
    features = feature_set[set_name]
    preprocessor = build_preprocessor(features["numeric"],features["cat"]  )

    metrics[set_name] = {}

    name = "random_forest"
    model = models[name]


    pipeline = Pipeline([
            ("preprocessor",preprocessor),
            ("model",model)

            ])

    cv = KFold(n_splits=5,shuffle=True,random_state=42)

    search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=30,scoring="neg_mean_absolute_error",cv = cv , random_state=42,n_jobs=1)



    search.fit(X_train, y_train)
    joblib.dump(search.best_estimator_,"models/regression/"+name+"_"+set_name+"_bestEst.joblib")
    bestParams = search.best_params_
    bestScore = -search.best_score_
    bestEstimator = search.best_estimator_
    bestIndex = int(search.best_index_)





    tuning_results = {
                "train_length": len(X_train),
                "feature_set": set_name,
                "id": timestamp,
                "CV_Folds" : cv.get_n_splits(),
                "best_index": bestIndex,
        "best_params": bestParams,
        "best_CV_MAE": bestScore,
        "best_CV_MAE_std": search.cv_results_["std_test_score"][bestIndex]
    }
    metrics[name] = tuning_results




    with open("results/regression/Tuning/"+timestamp+".json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
