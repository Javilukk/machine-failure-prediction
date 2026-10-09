from sklearn.preprocessing import OrdinalEncoder

def crear_feature_potencia(df):
    Rads_per_sec = df['Rotational speed [rpm]'] * (2 * 3.141592653589793 / 60)
    df['Potencia'] = df['Torque [Nm]'] * Rads_per_sec
    return df

def crear_feature_diferencia_temp(df):
    df['Diferencia_Temp'] = df['Process temperature [K]'] - df['Air temperature [K]']
    return df

def codificar_type(df):
    dataframe = OrdinalEncoder(categories=[['L', 'M', 'H']]).fit_transform(df[['Type']])
    df['Type_encoded'] = dataframe
    return df