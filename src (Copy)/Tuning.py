import duckdb
import numpy as np
import matplotlib.pyplot as plt 
import joblib
import json
import pandas as pd


from sklearn.pipeline import Pipeline

from taxi_ml.classifaction import prepare_data
from sklearn.model_selection import cross_validate,KFold

from sklearn.model_selection import RandomizedSearchCV

from datetime import datetime

from taxi_ml.common.preprocessing import build_preprocessor



con = duckdb.connect("data/taxi.db")






param_grid = {
    "model__n_estimators": [20,30,40,50,100],
    "model__max_depth": [5,8,10,20,None],
    "model__min_samples_leaf": [1,2,5,10],
    "model__max_features": [0.75,1.0,"sqrt"]
}



numerical_features = ["trip_seconds","trip_miles","hour_sin","hour_cos"]
categorical_features = ["weekday","company"]


feature_set = {}
feature_set["base_weekday_company"] = {
    "numeric": numerical_features,
    "cat": ["weekday", "company"]
}
evaluation_sets = {
    "all": None,
    "under_200": 200
}


X_train = pd.read_parquet("data/X_train.parquet")


y_train = pd.read_parquet("data/y_train.parquet")["trip_total"]


print(datetime.now())
metrics = {}

for set_name,features in feature_set.items():
    preprocessor = build_preprocessor(features["numeric"],features["cat"]  )

    metrics[set_name] = {}
    
    for name,model in [("random_forest", prepare_data["random_forest"])]:#models.items():
        
        
        pipeline = Pipeline([ 
        ("preprocessor",preprocessor),
        ("model",model)

        ])
        
        cv = KFold(n_splits=5,shuffle=True,random_state=42)

        search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=30,scoring="neg_mean_absolute_error",cv = cv , random_state=42,n_jobs=1)

        

        search.fit(X_train, y_train)
        joblib.dump(search.best_estimator_,"models/"+name+"_"+set_name+"_bestEst.joblib")


        bestParams = search.best_params_
        bestScore = -search.best_score_
        bestEstimator = search.best_estimator_
        bestIndex = int(search.best_index_)
        y_pred = search.predict(X_test)

        

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
        tuning_results = {
            "train_length": len(X_train),
            "test_length": len(X_test),
            "feature_set": set_name,
            "id": timestamp,
            "CV_Folds" : cv.get_n_splits(),
            "best_index": bestIndex,
    "best_params": bestParams,
    "best_CV_MAE": bestScore,
    "best_CV_MAE_std": search.cv_results_["std_test_score"][bestIndex]
}
        metrics[set_name][name] = tuning_results




with open("results/Tuning/"+timestamp+".json", "w") as f:
    json.dump(metrics, f, indent=2)

