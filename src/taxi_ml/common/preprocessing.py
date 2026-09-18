from sklearn.preprocessing import OneHotEncoder, StandardScaler


from sklearn.compose import ColumnTransformer


def build_preprocessor(numerical,categorical,scale_numeric =False ):
    if(scale_numeric):
        numerical_transformer = StandardScaler()
    else:
        numerical_transformer = "passthrough"
    result = ColumnTransformer( transformers = [ 
        ( "num", numerical_transformer , numerical),
("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),categorical )
                                                   ]
                                                   )
    return result
