import pandas as pd 

import json

import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit,RandomizedSearchCV
from datetime import datetime




from taxi_ml.common.preprocessing import build_preprocessor

from taxi_ml.forecasting.models import models
from taxi_ml.forecasting.config import feature_set

param_grid = {
    "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "model__max_iter": [30,50,80,100, 150],
    "model__max_leaf_nodes": [ 31, 63,95,127],
    "model__min_samples_leaf": [5,10, 15,20],
    "model__l2_regularization": [0.0, 0.1, 0.2,0.5],
    "model__max_features": [0.25,0.5, 0.75],
}
scoring_metrics = ["neg_mean_absolute_error", "r2"]

set_name = "lags_hour_weekday"
feature_list = feature_set[set_name]["numeric"]  + feature_set[set_name]["cat"] 
data_train = pd.read_parquet("data/forecasting/train_data.parquet")


X_train = data_train[feature_list]
y_train = data_train["trip_count"]

metrics = {}

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



features = feature_set[set_name]
preprocessor = build_preprocessor(features["numeric"],features["cat"]  )

model = models["gradient_boosting"]
name = "gradient_boosting"

#for name,model in [("HistGradientBoostingClassifier")]:
    
pipeline = Pipeline([ 
        ("preprocessor",preprocessor),
        ("model",model)

        ])
        
cv = TimeSeriesSplit(gap = 0, n_splits=5,test_size= 168*4)
search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=30,scoring="neg_mean_absolute_error",cv = cv , random_state=42,n_jobs=1)

        

search.fit(X_train, y_train)
joblib.dump(search.best_estimator_,"models/forecasting/"+name+"_"+set_name+"_bestEst.joblib")


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
    "best_CV_MAE": -bestScore,
    "best_CV_MAE_std": search.cv_results_["std_test_score"][bestIndex]
}
metrics[name] = tuning_results




with open("results/forecasting/Tuning/"+timestamp+".json", "w") as f:
    json.dump(metrics, f, indent=2)
