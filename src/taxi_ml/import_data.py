from pathlib import Path
import duckdb


def main():
    Path('data').mkdir(parents=True, exist_ok=True)

    con = duckdb.connect("data/taxi.db")

    con.sql("""
        CREATE TABLE IF NOT EXISTS raw_trips AS
        SELECT *
        FROM read_csv(
            'data/Taxi_2023.csv',
            thousands=','
        );
    """)
    project_root = Path(__file__).resolve().parents[2]
    text = (project_root / "sql" / "transform.sql").read_text()
    con.execute(text)
    con.close()


if __name__ == "__main__":
    main()
