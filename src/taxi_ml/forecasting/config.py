numerical_features = [f"lag_{x}" for x in range(1, 25)]
numerical_features += ["lag_168", "lag_336", "lag_504"]
hour_features = ["hour_sin","hour_cos"]
categorical_features = ["weekday"]
feature_set = {}
feature_set["lags"] = { "numeric" : numerical_features , "cat" : [] } 
feature_set["lags_weekday"] = { "numeric" : numerical_features  , "cat" : ["weekday"] } 
feature_set["lags_hour_weekday"] = { "numeric" : numerical_features +hour_features, "cat" : ["weekday"] } 
