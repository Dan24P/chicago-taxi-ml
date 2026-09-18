import numpy as np 
import matplotlib.pyplot as plt 
import json 


with open("results/regression/model_selection.json", 'r') as f:
    results = json.load(f)

models = results["base_company"].keys()

print(models)
first_feature_set = next(iter(results.values()))
first_model = next(iter(first_feature_set.values()))
metrics = first_model.keys()
metrics = [
    metric
    for metric in first_model.keys()
    if metric.startswith("test_")
]



for metric in metrics:
    fig,ax = plt.subplots()
    x = 0.5
    for model in models:
        x = x + 1.0       
        mean = results["base_weekday_company"][model][metric]["mean"]
        std = results["base_weekday_company"][model][metric]["std"]
        ax.errorbar(x,mean,yerr = std,label = model,fmt = "o")

    fig.legend()
    fig.suptitle(metric)
    fig.savefig("results/regression/plots/"+metric+".pdf")
