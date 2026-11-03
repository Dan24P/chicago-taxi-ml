import duckdb
import numpy as np 
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse,os
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size",type = int,default=os.getenv("SAMPLE_SIZE", "100000"))
    parser.add_argument(
        "--data-mode",
        choices=["sample", "full"],
        default=os.getenv("DATA_MODE", "sample"),
    )
    args = parser.parse_args()
    if args.data_mode not in ("sample", "full"):
        parser.error("Data mode must be 'sample' or 'full'")
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")

    output_dir = Path("data/classification")
    output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect("data/taxi.db")
    con.execute("SET threads = 1")
    sample_clause = ""

    if args.data_mode == "sample":
        sample_clause = (
        f"USING SAMPLE reservoir({args.sample_size} ROWS) REPEATABLE(42)"
    )

    df = con.sql(f"""
        SELECT
            trip_seconds,
            trip_total,
            trip_miles,
            EXTRACT(HOUR FROM trip_start) AS hour,
            EXTRACT(DOW FROM trip_start) AS weekday,
            company
        FROM (
        SELECT *
        FROM trips
        WHERE trip_total > 0
          AND trip_seconds > 0
          AND trip_miles > 0
          ) AS cleaned
          {sample_clause}
    """).df()
    con.close()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["high_fare"] = df["trip_total"] >= 50



    numerical_features = ["trip_seconds","trip_miles","hour_sin","hour_cos"]
    categorical_features = ["weekday","company"]


    X = df[numerical_features+categorical_features]
    y = df["high_fare"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
            stratify=y,
    )



    X_train.to_parquet(output_dir / "X_train.parquet")
    X_test.to_parquet( output_dir / "X_test.parquet")

    y_train.to_frame("high_fare").to_parquet("data/classification/y_train.parquet")
    y_test.to_frame("high_fare").to_parquet("data/classification/y_test.parquet")

if __name__=="__main__":
    main()