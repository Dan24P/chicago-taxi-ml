import subprocess
import sys



types = ["classifaction","regression"]

scripts = [
    "prepare_data",
    "model_selection",
    "tune_model",
    "final_evaluation"
]


for type in types:
    print("Running :", type)
    for script in scripts:
        path = "taxi_ml."+type+"."+script
        print(path)
        subprocess.run(
        [sys.executable, "-m",path],
        check=True
    )