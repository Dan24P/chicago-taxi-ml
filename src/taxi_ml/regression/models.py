
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor




models = {
    "linear": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=50,
        max_depth=10,
        min_samples_leaf=10,
        n_jobs=8,
        random_state=42),
    "gradient_boosting": HistGradientBoostingRegressor()
}