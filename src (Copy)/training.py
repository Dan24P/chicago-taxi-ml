import duckdb
import numpy as np
import matplotlib.pyplot as plt 
import joblib
import json
import pandas


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from taxi_ml.classifaction import prepare_data
from sklearn.model_selection import KFold

from sklearn.model_selection import RandomizedSearchCV



from taxi_ml.common.preprocessing import build_preprocessor



con = duckdb.connect("data/taxi.db")



df = con.sql("""
    SELECT
        trip_seconds,
        trip_total,
        trip_miles,
        EXTRACT(HOUR FROM trip_start) AS hour,
        EXTRACT(DOW FROM trip_start) AS weekday,
        company
    FROM trips
    WHERE trip_total > 0
      AND trip_seconds > 0
      AND trip_miles > 0
      USING SAMPLE 200000 ROWS
""").df()




param_grid = {
    "model__n_estimators": [50,100,150],
    "model__max_depth": [10,20,30,None],
    "model__min_samples_leaf": [1,5,10,20],
    "model__max_features": [0.5,1.0,"sqrt"]
}


df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)


numerical_features = ["trip_seconds","trip_miles","hour_sin","hour_cos"]
categorical_features = ["weekday","company"]


feature_set = {}
feature_set["base"] = { "numeric" : numerical_features , "cat" : [] } 
feature_set["base_weekday"] = { "numeric" : numerical_features , "cat" : ["weekday"] } 
feature_set["base_company"] = { "numeric" : numerical_features , "cat" : ["company"] } 
feature_set["base_weekday_company"] = {
    "numeric": numerical_features,
    "cat": ["weekday", "company"]
}
evaluation_sets = {
    "all": None,
    "under_200": 200
}

X = df[
    numerical_features+categorical_features
]

y = df["trip_total"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)




metrics = {}

validation_metrics = {}
for set_name,features in feature_set.items():
    preprocessor = build_preprocessor(features["numeric"],features["cat"]  )
    #X_transformed = preprocessor.fit_transform(X_train)


    metrics[set_name] = {}
    validation_metrics[set_name]= {}


    for name,model in [("random_forest", prepare_data["random_forest"])]:#models.items():
        metrics[set_name][name] = {}
        
        pipeline = Pipeline([ 
        ("preprocessor",preprocessor),
        ("model",model)

        ])
        
        cv = KFold(n_splits=5,shuffle=True,random_state=42)
       # scoring = cross_validate( pipeline,X= X_train, y = y_train ,  cv = cv ,n_jobs = 2,scoring=["neg_mean_absolute_error", "r2"])

        #mean_abs_err = -scoring["test_neg_mean_absolute_error"]

        #test_r2 = scoring["test_r2"]

        #validation_metrics[set_name][name] = {
        #    "MAE": mean_abs_err.mean(),
        #    "MAE_std" :mean_abs_err.std(),
        #    "r2": test_r2.mean(),
        #    "r2_std":test_r2.std()
        #}

        search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=10,scoring="neg_mean_absolute_error",cv = cv , random_state=42,n_jobs=4)

        

        search.fit(X_train, y_train)
        joblib.dump(search.best_estimator_,"models/"+name+"_"+set_name+".joblib")


        print(search.best_params_)
        print(search.best_score_)
        print(search.best_estimator_)
        y_pred = search.predict(X_test)


        y_pred_train = search.predict(X_train)
        for eval_name,val in evaluation_sets.items():


            if val is None:
                mask_test = y_test == y_test
                mask_train = y_train == y_train
            else:
                mask_test = y_test < val
                mask_train = y_train < val



            MAE_val = mean_absolute_error(y_test[mask_test], y_pred[mask_test])
            r2_Val = r2_score(y_test[mask_test], y_pred[mask_test])
            MAE_val_train = mean_absolute_error(y_train[mask_train], y_pred_train[mask_train])
            r2_Val_train = r2_score(y_train[mask_train], y_pred_train[mask_train])

            metrics[set_name][name][eval_name] = {
                "MAE": MAE_val,
                "R2" : r2_Val,
            }

    

results = X_test.copy()

X_train.to_parquet("data/X_train.parquet")
X_test.to_parquet("data/X_test.parquet")
y_test.to_frame("trip_total").to_parquet("data/y_test.parquet")




with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open("results/validation_metrics.json", "w") as f:
    json.dump(validation_metrics, f, indent=2)