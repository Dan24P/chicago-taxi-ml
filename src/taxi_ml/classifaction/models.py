from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier



models = {
    "logistic":{
        "model": LogisticRegression(max_iter= 500),
        "scale": True},
    "random_forest":{
        "model": RandomForestClassifier(
        n_jobs=8,
        random_state=42),
    "scale": False},
    "DecisionTreeClassifier":{
        "model": DecisionTreeClassifier(random_state=42

        ),
        "scale":False},    
    "gradient_boosting":{
        "model": HistGradientBoostingClassifier(random_state=42
),
                         "scale":False}
}