import subprocess
import sys
import argparse,os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices = ["classification","regression","forecasting"],
        default = "classification",
    )
    parser.add_argument(
    "--data-mode",
    choices=["sample", "full"],
    default=os.getenv("DATA_MODE", "sample"),
)
    parser.add_argument(
        "--sample-size",type= int,default=os.getenv("SAMPLE_SIZE", "100000")
    )
    args = parser.parse_args()
    if args.data_mode not in ("sample", "full"):
        parser.error("Data mode must be 'sample' or 'full'")
    types = [args.task]
    scripts = [
        "prepare_data",
        "model_selection",
        "tune_model",
        "final_evaluation"
    ]


    if args.task == "classification":
        scripts.append("model_analysis")

    for type in types:
        print("Running :", type, flush=True)
        for script in scripts:
            path = "taxi_ml."+type+"."+script
            command = [sys.executable, "-m",path]
            if script == "prepare_data" and type in ("regression","classification"):
                command.extend(["--sample-size",str(args.sample_size),"--data-mode",str(args.data_mode)])
                #command.append(str(args.sample_size))
            
            print(path, flush=True)
            subprocess.run(
            command,
            check=True
        )


if __name__ == "__main__":
    main()
