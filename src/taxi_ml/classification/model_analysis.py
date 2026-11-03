import joblib
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import json 
from sklearn.metrics import average_precision_score,brier_score_loss
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve,CalibrationDisplay

from pathlib import Path




def main():
    output_dir = Path("results/classification/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load("models/classification/gradient_boosting_base_company_bestEst.joblib")

    y_test = pd.read_parquet("data/classification/y_test.parquet")["high_fare"]
    y_train = pd.read_parquet("data/classification/y_train.parquet")["high_fare"]
    X_test = pd.read_parquet("data/classification/X_test.parquet")






    y_prob = model.predict_proba(X_test)[:,1]






    Nrepeats = 30
    randomState = 42 

    permut_test = permutation_importance(model,X_test,y_test,scoring="average_precision",n_repeats =Nrepeats,random_state=randomState )




    feature_names = X_test.columns

    output = {}

    fig,ax = plt.subplots()







    zipped_importance = zip(feature_names, permut_test.importances_mean.tolist(),permut_test.importances_std.tolist())

    importance_sorted = sorted(zipped_importance, key = lambda x : x[1] )

    for feature, mean, std in importance_sorted:
        output[feature] ={
            "mean": mean,
            "std": std
        }
    with open("results/classification/Importance.json", "w") as f:
        json.dump(output, f, indent=2)

    ##################### Feature Importance ##########################


    features = [x[0] for x in importance_sorted if x[0] != "weekday"]
    means = [x[1] for x in importance_sorted if x[0] != "weekday"]
    stds = [x[2] for x in importance_sorted if x[0] != "weekday"]

    ax.barh(features,width = means ,xerr = stds,capsize = 3,height= 0.75)

    ax.set_xlabel("Decrease in Average Precision")
    ax.set_title("Permutation Feature Importance")

    fig.tight_layout()
    fig.savefig("results/classification/plots/importance.pdf")
    plt.close()


    ################ Calibration curve #####################




    prob_true,prob_pred = calibration_curve(y_test,y_prob,n_bins=10)

    fig,ax = plt.subplots()
    disp = CalibrationDisplay(prob_true,prob_pred,y_prob )
    disp.plot(ax = ax)
    plt.tight_layout()
    fig.savefig("results/classification/plots/calibration.pdf")

    plt.close()



    ############ Calculate brier score and compare to baseline 

    brier_score = brier_score_loss(y_test,y_prob)




    y_train_mean = float(y_train.mean())

    baseline_prob =  np.full(len(y_test), y_train_mean )
    brier_score_base = brier_score_loss(y_test,baseline_prob)



    output = {
        "number of repeats": Nrepeats,
        "random state": randomState,
        "brier score ": brier_score,
        "brier score baseline": brier_score_base,
        "baseline_probability": y_train_mean,
        "positive_rate": float(y_test.mean())
    }

    with open("results/classification/brier_score.json", "w") as f:
        json.dump(output, f, indent=2)
if __name__=="__main__":
    main()