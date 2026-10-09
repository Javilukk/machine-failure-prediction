from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from features import crear_feature_potencia, crear_feature_diferencia_temp, codificar_type
import pandas as pd

app = FastAPI()

modelo = joblib.load('RandomForestMachineFailure.pkl')

class DatosSensor(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    type: str
    pass

@app.post("/predict")
def predecir(datos: DatosSensor):
    datos_dict= {
        'Air temperature [K]': [datos.air_temperature],
        'Process temperature [K]': [datos.process_temperature],
        'Rotational speed [rpm]': [datos.rotational_speed],
        'Torque [Nm]': [datos.torque],
        'Tool wear [min]': [datos.tool_wear],
        'Type': [datos.type]
    }


    pandas_df = pd.DataFrame(datos_dict)
    pandas_df = crear_feature_potencia(pandas_df)
    pandas_df = crear_feature_diferencia_temp(pandas_df)
    pandas_df = codificar_type(pandas_df)
    pandas_df = pandas_df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Diferencia_Temp', 'Potencia', 'Type_encoded']]
    prediccion = modelo.predict_proba(pandas_df)
    return {"probabilidad_fallo": float(prediccion[0][1])}
    pass
