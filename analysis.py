import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 




con = duckdb.connect("taxi.db")


con.sql(""" DESCRIBE trips """).show()

con.sql("""
SELECT company, AVG(trip_total) AS avg_revenue, AVG(trip_seconds) AS avg_duration
FROM trips
GROUP BY company
""").show()


df = con.sql("""
SELECT company, AVG(trip_total) AS avg_revenue, AVG(trip_seconds) AS avg_duration
FROM trips
GROUP BY company
""").df()


fit = np.polyfit(df["avg_duration"], df["avg_revenue"], 1)

xplot = np.linspace(df["avg_duration"].min(), df["avg_duration"].max(), 100)
yplot = fit[0] * xplot + fit[1]


plt.bar(df["company"], df["avg_revenue"])

plt.xlabel("Company")
plt.ylabel("Average Revenue in $")
plt.title("Average Revenue per Trip by Company")
plt.savefig("average_revenue_per_trip_by_company.pdf")
plt.close()

plt.plot(xplot, yplot, '-')
plt.plot(df["avg_duration"], df["avg_revenue"], 'o')
plt.xlabel("Average Duration in Seconds")
plt.ylabel("Average Revenue in $")
plt.title("Average Revenue vs Average Duration by Company")
plt.savefig("average_revenue_vs_average_duration_by_company.pdf")
plt.close()




df = con.sql("""
SELECT AVG(trip_total) as avg_revenue,EXTRACT(MONTH FROM trip_start) AS month, COUNT(*) AS trip_count FROM trips GROUP BY month;
""").df()



plt.bar(df["month"], df["avg_revenue"])
plt.xlabel("Month")
plt.ylabel("Average Revenue in $")
plt.title("Average Revenue per Trip by Month")
plt.savefig("average_revenue_per_trip_by_month.pdf") 
plt.close()




plt.bar(df["month"], df["trip_count"])
plt.xlabel("Month")
plt.ylabel("Number of Trips")
plt.title("Number of Trips by Month")
plt.savefig("number_of_trips_by_month.pdf")
plt.close()
