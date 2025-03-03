import pandas as pd

def label_encoding(df, column):
    
    df[column + '_Encoded'] = pd.factorize(df[column])[0]
    return df
    
    
def one_hot_encoding(df, column):
  
    encoded = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, encoded], axis=1)
    return df