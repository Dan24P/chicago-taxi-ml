import matplotlib.pyplot as plt 
import joblib
import pandas as pd
import json




plots = { 
    "mean_residual":"residual",
    "mean_rel_error":"relative_error",
    "mae":"mae"
}

binned_results = {}

with open("results/metrics.json", "r") as f:
    metrics = json.load(f)



X_test = pd.read_parquet("data/X_test.parquet")
y_test = pd.read_parquet("data/y_test.parquet")["trip_total"]



for name in metrics:
        model = joblib.load("models/"+name+".joblib")



        y_pred = model.predict(X_test)


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
        binned_results[name] = binned

for plot, label in plots.items():



    for name in metrics.keys():
        binned = binned_results[name]
        plt.plot(binned["mean_data"] , binned[plot] ,"o",label = label )

    plt.xlabel(" mean bin ")
    plt.ylabel(label)
    plt.legend()
    plt.savefig("plots/"+plot+".png")
    plt.close()

