
import joblib
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import json 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix

def ThresholdScan(y_prob,threshold):
    result = y_prob >= threshold
    return result


def Metric_threshold(metric,y_prob,y_test,threshold):
    y_threshold = ThresholdScan(y_prob,threshold)
    result = metric( y_test,y_threshold )
    return result


def ConfusinMatThreshold(y_prob,y_test,threshold):
    y_threshold = ThresholdScan(y_prob,threshold)
    result = confusion_matrix(y_true=y_test,y_pred=y_threshold)
    return result


scoring_metrics = {
    "accuracy": accuracy_score,
    "precision":precision_score,
    "recall":recall_score,
    "f1":f1_score}

model = joblib.load("models/gradient_boosting_base_company_bestEst.joblib")

y_test = pd.read_parquet("data/classification/y_test.parquet")["high_fare"]

X_test = pd.read_parquet("data/classification/X_test.parquet")




y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]


#MAE_val = mean_absolute_error(y_test, y_pred_rf)
#R2_val = r2_score(y_test, y_pred_rf)

acc_val =      accuracy_score(y_test,y_pred)
pre_val =      precision_score(y_test,y_pred)
rec_val =      recall_score(y_test,y_pred)
f1_val =       f1_score(y_test,y_pred)
roc_auc_val =  roc_auc_score(y_test,y_prob)
avg_prec_val = average_precision_score(y_test,y_prob)



metrics = {
            "accuracy": acc_val,
            "precision": pre_val,
            "recall": rec_val,
            "f1_score": f1_val,
            "roc_auc": roc_auc_val,
            "average precision": avg_prec_val
 }


with open("results/classification/final_evaluation.json", "w") as f:
    json.dump(metrics, f, indent=2)


results = {
}

ThresholdVals = np.linspace(0,1,100)
for name,metric in scoring_metrics.items():
    
    results[name] = []
    #MetricVal = Metric_threshold(metric,y_prob,y_test,ThresholdVals)
    for val in ThresholdVals:
       results[name].append( Metric_threshold(metric,y_prob,y_test,val))

fig, ax = plt.subplots(2, 2, figsize=(10, 7))
axes = ax.ravel()

for axis, metric in zip(axes, scoring_metrics):
    print(metric)
    axis.plot(ThresholdVals, results[metric])

    axis.set_title(metric)
    axis.set_ylabel(metric)
    axis.set_xlabel("Threshold")

    maxInd = np.argmax(results[metric])
    print("maxThreshold",ThresholdVals[maxInd])
    print("maxVal",results[metric][maxInd])
fig.tight_layout()

plt.savefig("results/classification/plots/threshold_metrics.pdf")
plt.close()

results["TN"] = []
results["FP"] = []
results["FN"] = []
results["TP"] = []
for val in ThresholdVals:
       tn, fp, fn, tp = ConfusinMatThreshold(y_prob,y_test,val).ravel()
       results["TN"].append( tn )
       results["FP"].append( fp )
       results["FN"].append( fn )
       results["TP"].append( tp )



events = ["TN","FP","FN","TP"]

fig, ax = plt.subplots(2, 2, figsize=(10, 7))
axes = ax.ravel()

for axis,event in zip(axes,events):
    axis.plot(ThresholdVals,results[event])

    axis.set_title(event)
    axis.set_ylabel(event)
    axis.set_xlabel("Threshold")

fig.tight_layout()

plt.savefig("results/classification/plots/classification_outcomes.pdf")
plt.close()
