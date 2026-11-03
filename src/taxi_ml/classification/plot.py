import numpy as np 
import matplotlib.pyplot as plt 
import json 
from pathlib import Path 
def main():
    output_dir = Path("results/classification/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open("results/classification/model_selection.json", 'r') as f:
        results = json.load(f)

    models = results["base_company"].keys()


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
            mean = results["base_company"][model][metric]["mean"]
            std = results["base_company"][model][metric]["std"]
            ax.errorbar(x,mean,yerr = std,label = model,fmt = "o")

        fig.legend()
        fig.suptitle(metric)
        fig.savefig("results/classification/plots/"+metric+".pdf")
        plt.close(fig)
if __name__=="__main__":
    main()