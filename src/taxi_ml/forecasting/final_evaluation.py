from pathlib import Path
import joblib
import pandas as pd
import json
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from  taxi_ml.forecasting.config import feature_set


def main():
    Path('results/forecasting/plots').mkdir(parents=True, exist_ok=True)

    model = joblib.load("models/forecasting/gradient_boosting_lags_hour_weekday_bestEst.joblib")

    data_test = pd.read_parquet("data/forecasting/test_data.parquet")

    set_name = "lags_hour_weekday"
    feature_list = feature_set[set_name]["numeric"]  + feature_set[set_name]["cat"]
    X_test = data_test[feature_list]
    y_test = data_test["trip_count"]



    y_pred = model.predict(X_test)

    data_test["prediction"] = y_pred


    MAE_val = mean_absolute_error(y_test, y_pred)
    R2_val = r2_score(y_test, y_pred)


    metrics = {
        "MAE": float(MAE_val),
        "r2":  float(R2_val)
     }


    with open("results/forecasting/final_evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)


    data_test["residuals"] = y_pred - y_test

    timestamp = data_test["timestamp"]


    results = data_test[["timestamp", "trip_count"]].copy()
    results["pred"] = y_pred
    results["residual"] = results["pred"] - results["trip_count"]

    fig,ax = plt.subplots()

    ax.plot(results["timestamp"],results["pred"],label = "predicted")
    ax.plot(results["timestamp"],results["trip_count"],label = "data")

    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xlabel("timestamp")
    ax.set_ylabel("number of trips")

    plt.legend()
    plt.tight_layout()
    plt.savefig("results/forecasting/plots/pred_vs_data.pdf")
    plt.close()


    fig,ax = plt.subplots()

    ax.plot(timestamp[:30],data_test["residuals"][:30],label = "residuals")

    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xlabel("timestamp")
    ax.set_ylabel("number of trips")
    plt.legend()
    plt.tight_layout()

    plt.savefig("results/forecasting/plots/residuals_over_time.pdf")
    plt.close()
    plots = ["month","weekday"]





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
                mean_pred = ("pred","mean"),
            )
    binned_results = binned

    fig,ax  = plt.subplots(2,1)


    ax[0].plot(binned["mean_data"] , binned["mean_data"] ,"o" ,label = "data")
    ax[0].plot(binned["mean_data"] , binned["mean_pred"] ,"o" ,label = "pred")
    ax[0].set_xlabel(" Mean predicted trips per bin")
    ax[0].set_ylabel("mean numer of trips")




    ax[1].plot(binned["mean_data"] , binned["mean_residual"] ,"o" )
    ax[1].set_xlabel(" Mean predicted trips per bin")
    ax[1].set_ylabel("mean residual")
    fig.tight_layout()
    fig.legend()
    fig.savefig("results/forecasting/plots/residuals.pdf")
    plt.close(fig)

    fig,ax  = plt.subplots()

    ax.plot(binned["mean_data"],binned["mae"],"o")
    ax.set_xlabel(" binned predicitons")
    ax.set_ylabel("mean absolute error")
    fig.tight_layout()
    fig.savefig("results/forecasting/plots/mae.pdf")
    plt.close(fig)


    fig,ax  = plt.subplots()

    ax.plot(binned["mean_data"],binned["mean_rel_error"],"o")
    ax.set_xlabel(" Mean predicted trips per bin")
    ax.set_ylabel("Mean relative absolute error")
    fig.tight_layout()
    fig.savefig("results/forecasting/plots/rel_error.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
