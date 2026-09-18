
import joblib
import pandas as pd 
import json 
from sklearn.metrics import mean_absolute_error, r2_score





rf = joblib.load("models/random_forest_base_weekday_company_bestEst.joblib")

y_test = pd.read_parquet("data/regression/y_test.parquet")["trip_total"]

X_test = pd.read_parquet("data/regression/X_test.parquet")




y_pred_rf = rf.predict(X_test)

MAE_val = mean_absolute_error(y_test, y_pred_rf)
R2_val = r2_score(y_test, y_pred_rf)


metrics = {
    "MAE": float(MAE_val),
    "r2":  float(R2_val)
 }


with open("results/regression/final_evaluation.json", "w") as f:
    json.dump(metrics, f, indent=2)
