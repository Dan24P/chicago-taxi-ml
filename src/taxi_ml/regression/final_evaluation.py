
import joblib
import pandas as pd 
import json 
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt 




rf = joblib.load("models/regression/random_forest_base_weekday_company_bestEst.joblib")

y_test = pd.read_parquet("data/regression/y_test.parquet")["trip_total"]

X_test = pd.read_parquet("data/regression/X_test.parquet")




y_pred = rf.predict(X_test)

MAE_val = mean_absolute_error(y_test, y_pred)
R2_val = r2_score(y_test, y_pred)


metrics = {
    "MAE": float(MAE_val),
    "r2":  float(R2_val)
 }


with open("results/regression/final_evaluation.json", "w") as f:
    json.dump(metrics, f, indent=2)








results = pd.DataFrame({
            "pred":y_pred,
            "data":y_test}
        )

results["residual"] = results["pred"] - results["data"]
results["abs_error"] = results["residual"].abs()
results["rel_error"] = results["abs_error"] / results["data"]


results["bins"] = pd.qcut( results["data"],q = 10)
    
binned = results.groupby("bins", observed=True).agg(
            mean_data=("data", "mean"),
            mean_residual=("residual", "mean"),
            mean_rel_error=("rel_error", "mean"),
            mae=("abs_error", "mean"),
        )
binned_results = binned

fig,ax  = plt.subplots(2,1)


ax[0].plot(binned["mean_data"] , binned["mean_data"] ,"o" )
ax[0].set_xlabel(" mean price in dollar")
ax[0].set_ylabel("mean data")




ax[1].plot(binned["mean_data"] , binned["mean_residual"] ,"o" )
ax[1].set_xlabel(" mean price in dollar ")
ax[1].set_ylabel("mean residual")
fig.legend()
fig.savefig("results/regression/plots/residuals.pdf")
