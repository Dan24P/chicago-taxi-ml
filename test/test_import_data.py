import pytest 
from taxi_ml.import_data import main
from pathlib import Path
import duckdb,csv
import pandas as pd 
def test_import_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    f = open("data/Taxi_2023.csv","w")

    f.write("Trip ID,Trip Start Timestamp,Trip Miles,Company,Trip Total,Trip Seconds\n test-1,2023-01-01 12:00:00,3.0,Test Taxi,20.00,600")
    f.close()
    main()
    con = duckdb.connect("data/taxi.db", read_only=True)
    df = con.execute("SELECT * FROM raw_trips").df()

    assert len(df) == 1


    assert  df["Trip ID"][0]  == ' test-1'
    assert  df["Trip Start Timestamp"][0]  == pd.Timestamp(' 2023-01-01 12:00:00')
    assert  df["Trip Miles"][0]  ==  3.0
    assert  df["Company"][0]  == 'Test Taxi'
    assert  df["Trip Total"][0]  ==  20.00
    assert  df["Trip Seconds"][0]  ==  600


    df = con.execute("SELECT * FROM trips").df()
    
    assert len(df) == 1
    
    assert  df["trip_id"][0]  == ' test-1'
    assert  df["trip_start"][0]  == pd.Timestamp(' 2023-01-01 12:00:00')
    assert  df["trip_miles"][0]  ==  3.0
    assert  df["company"][0]  == 'Test Taxi'
    assert  df["trip_total"][0]  ==  20.00
    assert  df["trip_seconds"][0]  ==  600