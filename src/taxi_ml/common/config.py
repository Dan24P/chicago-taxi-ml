numerical_features = ["trip_seconds","trip_miles","hour_sin","hour_cos"]
categorical_features = ["weekday","company"]
feature_set = {}
feature_set["base"] = { "numeric" : numerical_features , "cat" : [] } 
feature_set["base_weekday"] = { "numeric" : numerical_features , "cat" : ["weekday"] } 
feature_set["base_company"] = { "numeric" : numerical_features , "cat" : ["company"] } 
feature_set["base_weekday_company"] = {
    "numeric": numerical_features,
    "cat": ["weekday", "company"]
}