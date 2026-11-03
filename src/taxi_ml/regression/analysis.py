from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import pandas as pd


def main():
    Path('results/regression/plots').mkdir(parents=True, exist_ok=True)

    plots = {
        "mean_residual":"residual",
        "mean_rel_error":"relative_error",
        "mae":"mae"
    }

    binned_results = {}

    model_paths = {
        "random_forest": "models/regression/random_forest_base_weekday_company_bestEst.joblib"
    }



    X_test = pd.read_parquet("data/regression/X_test.parquet")
    y_test = pd.read_parquet("data/regression/y_test.parquet")["trip_total"]



    for name, model_path in model_paths.items():
            model = joblib.load(model_path)



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



        for name in model_paths:
            binned = binned_results[name]
            plt.plot(binned["mean_data"] , binned[plot] ,"o",label = name )

        plt.xlabel(" mean bin ")
        plt.ylabel(label)
        plt.legend()
        plt.savefig("results/regression/plots/"+plot+".png")
        plt.close()


if __name__ == "__main__":
    main()
