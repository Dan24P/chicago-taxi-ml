import duckdb

con = duckdb.connect("taxi.db")

con.sql("""
    CREATE TABLE IF NOT EXISTS raw_trips AS
    SELECT *
    FROM read_csv(
        'data/Taxi_2023.csv',
        thousands=','
    );
""")