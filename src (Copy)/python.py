import duckdb 

con = duckdb.connect("taxi.db")

#con.sql("""
#    CREATE TABLE trips AS
#    SELECT *
#    FROM read_csv(
#        'Taxi_Trips_-_2023_20260907.csv',
#        thousands = ','
#    )
#""")

con.sql("""
    SELECT AVG("Trip Miles") FROM trips
""").show()


con.sql("""
    SELECT SUM("Trip Miles") FROM trips
""").show()

con.sql("""
    DESCRIBE trips
""").show()
#con.sql("""
#     ALTER TABLE trips ADD COLUMN trip_total_num DOUBLE;
#
#     UPDATE trips 
#     SET trip_total_num =
#        TRY_CAST(
#            REPLACE(REPLACE("Trip Total", '$', ''), ',', '')
#            AS DOUBLE
#        );
#     """)


con.sql("""
    SELECT  trip_total_num FROM trips LIMIT 10;
""").show()