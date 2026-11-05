import duckdb
from pathlib import Path 
import sys
from taxi_ml.classification.prepare_data import main
import pandas as pd
import pytest 


@pytest.fixture
def trips_database(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()

    with duckdb.connect("data/taxi.db") as con:
        con.execute(
            """
        CREATE TABLE trips AS
        SELECT 600.0 + i AS trip_seconds,
        3.0 + 2*i AS trip_miles,
        CASE WHEN i % 2 = 0 THEN 20.0 ELSE 60.0 END AS trip_total,
        TIMESTAMP '2023-01-01 12:00:00' AS trip_start, 
        'Test Taxi' AS company 
        FROM range(100) AS t(i)
    """
        )

def test_sample_size(trips_database,monkeypatch):
    monkeypatch.setattr(
        sys,"argv",["prepare_data.py","--data-mode","sample","--sample-size","50"],
    )
    main()

    X_train = pd.read_parquet("data/classification/X_train.parquet")
    y_train = pd.read_parquet("data/classification/y_train.parquet")

    assert len(X_train) == 0.8*50
    assert len(y_train) == 0.8*50

    X_test = pd.read_parquet("data/classification/X_test.parquet")
    y_test = pd.read_parquet("data/classification/y_test.parquet")

    assert len(X_test) == 0.2*50
    assert len(y_test) == 0.2*50

def test_full_mode(trips_database,monkeypatch):
    monkeypatch.setattr(
        sys,"argv",["prepare_data.py","--data-mode","full","--sample-size","50"],
    )
    main()

    X_train = pd.read_parquet("data/classification/X_train.parquet")
    y_train = pd.read_parquet("data/classification/y_train.parquet")

    assert len(X_train) == 0.8*100
    assert len(y_train) == 0.8*100

    X_test = pd.read_parquet("data/classification/X_test.parquet")
    y_test = pd.read_parquet("data/classification/y_test.parquet")

    assert len(X_test) == 0.2*100
    assert len(y_test) == 0.2*100


def test_invalid_trips_are_filtered(trips_database, monkeypatch):
    with duckdb.connect("data/taxi.db") as con:
            con.execute(
                """
            INSERT INTO trips (trip_seconds, trip_miles, trip_total, trip_start, company)
            VALUES
                (0, 10 , 10 ,TIMESTAMP '2023-01-01 12:00:00','Test Taxi'),
                (10,0,10,TIMESTAMP '2023-01-01 12:00:00','Test Taxi'),
                (10,10,0,TIMESTAMP '2023-01-01 12:00:00','Test Taxi')
        """
            )
    monkeypatch.setattr(
            sys,"argv",["prepare_data.py","--data-mode","full","--sample-size","50"],
        )
    main()

    X_train = pd.read_parquet("data/classification/X_train.parquet")
    y_train = pd.read_parquet("data/classification/y_train.parquet")

    assert len(X_train) == 0.8*100
    assert len(y_train) == 0.8*100

    X_test = pd.read_parquet("data/classification/X_test.parquet")
    y_test = pd.read_parquet("data/classification/y_test.parquet")

    assert len(X_test) == 0.2*100
    assert len(y_test) == 0.2*100