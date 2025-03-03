import pandas as pd
from sklearn.preprocessing import StandardScaler , MinMaxScaler

def scale_one_col(df , column , scaler ):
    df[f'{column}_scaled']= scaler.fit_transform(df[[column]])
    return df

def scale_all(df, scaler):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaled_data = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=[f'{col}_scaled' for col in numeric_cols])
    return pd.concat([df.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)
      
      