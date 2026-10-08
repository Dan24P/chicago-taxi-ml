import duckdb
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import pandas as pd
con = duckdb.connect("data/taxi.db")

def naive_forecasting(t,data):
    data = data.sort_values("timestamp")
    data["lag_1"] = data["trip_count"].shift(1)
    data["lag_24"] = data["trip_count"].shift(24)
    data["lag_168"] = data["trip_count"].shift(168)
    
    result = data[ data["timestamp"] == t]
    return result




df = con.sql("""
    SELECT 
        DATE_TRUNC('hour',trip_start) AS timestamp,
        EXTRACT ( HOUR FROM timestamp) AS hour,
        EXTRACT (DOW FROM timestamp) as weekday,
        EXTRACT (MONTH FROM timestamp) as month,
        EXTRACT (DAY FROM timestamp) as day,
        COUNT(*) AS trip_count
    FROM trips
    GROUP BY timestamp
    ORDER BY timestamp
""").df()
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)




plt.plot(df["timestamp"], df["trip_count"])

plt.ylabel("number of trips ")

plt.savefig("results/forecasting/plots/yearly_ntrips.pdf")

plt.close()

plots = ["weekday","month","hour"]
for member in plots:
    trip_count = df.groupby(member)["trip_count"].mean()


    plt.bar(trip_count.index, trip_count,width= 0.3)


    plt.ylabel("avg number of trips ")
    #plt.xticks([0,1,2,3,4,5,6],labels= ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"])
    plt.savefig("results/forecasting/plots/"+member+"_ntrips.pdf")

    plt.close()




########### Autocorrelation ######################

corr = []

lag_val = np.linspace(1,200,200, dtype = int)
print(lag_val)
corr.append(1)
for lag in lag_val:
    corr.append( df["trip_count"].autocorr(lag = lag).tolist() )

lag_val = np.insert(lag_val,0,0,axis = 0)

plt.stem(lag_val,corr, "o")
plt.vlines(24,-1,1,ls = "dotted",color = "black", lw = 0.8)
plt.vlines(168,-1,1,ls = "dotted",color = "black", lw = 0.8)

plt.xlabel("Lag / hours ")
plt.ylabel("Autocorrelation")

plt.xlim(0,200)
plt.ylim(-1,1)

plt.savefig("results/forecasting/plots/autocorrelation.pdf")
plt.close()





############### Create Lag vals ###############


Lag_list = [x for x in range(1, 25)]
Lag_list.append(168)
Lag_list.append(168*2)
Lag_list.append(168*3)


df = df.sort_values("timestamp")

for lag in Lag_list:
    df["lag_"+str(lag)] = df["trip_count"].shift(lag)


df = df.dropna()
df = df.sort_values("timestamp")







split_1 = int( round(0.7 * len(df),0)) 
split_2 = int(round(0.85*len(df),0) )

train = df.iloc[:split_1]

validation = df.iloc[split_1:split_2]

test = df.iloc[split_2:]



train.to_parquet("data/forecasting/train_data.parquet")
validation.to_parquet("data/forecasting/validation_data.parquet")
test.to_parquet("data/forecasting/test_data.parquet")

#X_train = train[train.columns[8:]]
#print(X_train)
#y_train = train["trip_count"]
#print(y_train)
#
#X_validation = validation[validation.columns[8:]]
#y_validation = validation["trip_count"]
#
#X_test = test[test.columns[8:]]
#y_test = test["trip_count"]
#
#
#X_train.to_parquet("data/forecasting/X_train.parquet")
#X_validation.to_parquet("data/forecasting/X_validation.parquet")
#X_test.to_parquet("data/forecasting/X_test.parquet")
#
#y_train.to_frame("trip_count").to_parquet("data/forecasting/y_train.parquet")
#y_validation.to_frame("trip_count").to_parquet("data/forecasting/y_validation.parquet")
#y_test.to_frame("trip_count").to_parquet("data/forecasting/y_test.parquet")