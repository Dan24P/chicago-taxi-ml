CREATE OR REPLACE TABLE trips AS
SELECT
    "Trip ID" AS trip_id,
    "Trip Start Timestamp" AS trip_start,
    "Trip Miles" AS trip_miles,
    "Company" AS company,
    TRY_CAST(
        REPLACE(REPLACE(CAST("Trip Total" AS VARCHAR), '$', ''), ',', '')
        AS DOUBLE
    ) AS trip_total,
    TRY_CAST(
                REPLACE(REPLACE(CAST("Trip seconds" AS VARCHAR), '$', ''), ',', '')
                AS DOUBLE
            ) AS trip_seconds
FROM raw_trips;