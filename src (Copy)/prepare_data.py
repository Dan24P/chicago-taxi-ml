import duckdb
import numpy as np
from sklearn.model_selection import train_test_split




con = duckdb.connect("data/taxi.db")



df = con.sql("""
    SELECT
        trip_seconds,
        trip_total,
        trip_miles,
        EXTRACT(HOUR FROM trip_start) AS hour,
        EXTRACT(DOW FROM trip_start) AS weekday,
        company
    FROM trips
    WHERE trip_total > 0
      AND trip_seconds > 0
      AND trip_miles > 0
      USING SAMPLE 100000 ROWS
""").df()




df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)


numerical_features = ["trip_seconds","trip_miles","hour_sin","hour_cos"]
categorical_features = ["weekday","company"]




X = df[
    numerical_features+categorical_features
]

y = df["trip_total"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train.to_parquet("data/X_train.parquet")
X_test.to_parquet("data/X_test.parquet")

y_train.to_frame("trip_total").to_parquet("data/y_train.parquet")
y_test.to_frame("trip_total").to_parquet("data/y_test.parquet")