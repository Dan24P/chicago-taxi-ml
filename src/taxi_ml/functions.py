from sklearn.preprocessing import OneHotEncoder
from taxi_ml.classifaction import prepare_data


from sklearn.compose import ColumnTransformer


def build_preprocessor(numerical,categorical):
    result = ColumnTransformer( transformers = [ ( "num", "passthrough" , numerical),
                                                   ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),categorical )
                                                   ]
                                                   )
    return result
