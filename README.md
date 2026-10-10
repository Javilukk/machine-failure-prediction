# Machine Failure Prediction

Predicción de riesgo de falla en maquinaria industrial a partir de datos de sensores (temperatura, torque, velocidad rotacional, desgaste de herramienta), usando modelos de clasificación con interpretabilidad vía SHAP, expuesto como API con FastAPI y contenerizado con Docker.

## El problema

Dado un conjunto de lecturas de sensores de una máquina (fresadora industrial), predecir la probabilidad de que ocurra una falla. El dataset usado es **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository) — un dataset sintético de 10,000 instancias que simula condiciones realistas de mantenimiento industrial, con 3.39% de casos de falla (clase fuertemente desbalanceada).

Fuente: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

## Exploración de datos (EDA)

El dataset no tiene valores nulos. La variable objetivo (`Machine failure`) está desbalanceada: 96.61% no-falla / 3.39% falla.

**Hallazgo clave:** la variable `Torque [Nm]` muestra una distribución **bimodal** en la clase de falla — las fallas se concentran tanto en valores bajos (5-15 Nm) como altos (50-70 Nm) de torque, con la clase "no falla" ocupando la región central. Esto se explica por el mecanismo de falla `PWF` (Power Failure), que depende de la potencia (torque × velocidad angular) estando fuera de un rango aceptable — tanto por exceso como por defecto.

## Feature engineering

- **Potencia** = Torque [Nm] × velocidad angular (rpm convertido a rad/s), con base en el mecanismo físico detrás de PWF.
- **Diferencia_Temp** = Process temperature − Air temperature, con base en el mecanismo de `HDF` (Heat Dissipation Failure).
- **Type_encoded**: ordinal encoding (L=0, M=1, H=2) de la variable `Type`, ya que representa una jerarquía real de calidad del producto (Low < Medium < High).
- **Columnas excluidas (prevención de data leakage):** `UDI`, `Product ID` (identificadores sin relación causal) y `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (columnas de tipo de falla específico, que no estarían disponibles en el momento real de predicción).

## Train/test split

Split estratificado 70/30 (no 80/20), justificado por el tamaño reducido de la clase minoritaria (339 casos totales) — un test set más grande da una evaluación más confiable sobre la clase falla, a costa de menos ejemplos para entrenar.

## Resultados — comparación de modelos

| Modelo | Precision (falla) | Recall (falla) | F1 (falla) | Accuracy |
|---|---|---|---|---|
| DummyClassifier (baseline) | 0.00 | 0.00 | 0.00 | 0.97 |
| Logistic Regression | 0.57 | 0.20 | 0.29 | 0.97 |
| **Random Forest** | **0.95** | **0.78** | **0.86** | **0.99** |
| XGBoost (sin scale_pos_weight) | 0.85 | 0.79 | 0.82 | 0.99 |
| XGBoost (con scale_pos_weight ≈ 28.5) | 0.78 | 0.80 | 0.79 | 0.99 |

El DummyClassifier confirma que accuracy no es una métrica útil aquí: con 96.6% de desbalance, un modelo que no aprende nada logra 97% de accuracy sin detectar una sola falla real. Por eso la evaluación se centra en precision, recall y F1 de la clase minoritaria.

**Logistic Regression** mejora sobre el Dummy pero su recall (0.20) es limitado porque solo puede trazar una frontera de decisión lineal, incapaz de separar bien el patrón bimodal de Torque.

**Random Forest** supera claramente a Logistic Regression al poder capturar relaciones no-lineales mediante múltiples cortes por variable.

**XGBoost**, con o sin ajuste de peso por clase, no logró superar a Random Forest en F1 — el ajuste de `scale_pos_weight` mejora el recall marginalmente pero a un costo alto en precision.

**GridSearchCV** sobre Random Forest arrojó un resultado peor al modelo default (F1 0.70 vs 0.86) — pendiente de diagnóstico, probablemente relacionado con el uso de `class_weight='balanced'` en la combinación ganadora.

### Modelo final: Random Forest (parámetros default)

Se eligió Random Forest como modelo final por tener el mejor balance (F1) entre precision y recall, logrando un recall casi idéntico a las alternativas de XGBoost pero con muchas menos falsas alarmas.

## Análisis de errores (falsos negativos)

De las 102 fallas reales en test, Random Forest no detectó 22 (recall 0.78). Al comparar sus estadísticas contra los 80 correctamente detectados:

- **Torque:** los falsos negativos tienen mediana de 44.2 Nm (zona de traslape entre clases), frente a 55.35 Nm en los detectados (zona de separación clara).
- **Tool wear:** los falsos negativos muestran valores consistentemente altos (mediana 207 min) frente a mayor dispersión en los detectados (mediana 140.5 min).

**Conclusión:** el modelo falla principalmente en casos "ambiguos", donde los valores de sensor no caen en las regiones claramente asociadas a falla. Esto sugiere una limitación de información disponible en las variables actuales, no un problema de configuración del modelo.

## Interpretabilidad (SHAP)

### Importancia global de variables

Orden de importancia (mayor a menor impacto promedio en la predicción): **Rotational speed > Tool wear > Torque > Diferencia_Temp > Potencia > Air temperature > Process temperature > Type_encoded**.

Un hallazgo relevante: `Rotational speed` resultó ser la variable más importante, por encima de `Torque`, pese a que el EDA inicial se enfocó en el patrón bimodal de Torque. `Rotational speed` muestra un patrón similar (bimodal), consistente con su rol conjunto en el cálculo de potencia (mecanismo de PWF).

`Diferencia_Temp` muestra una relación contraintuitiva: valores **bajos** empujan hacia falla, no altos. Esto es consistente con el mecanismo físico de `HDF`: si el calor generado por el proceso no logra disipar correctamente hacia el ambiente, la diferencia de temperatura proceso-ambiente se mantiene baja (el calor queda atrapado), en vez de alta.

### Análisis local (caso individual)

Se analizó un caso específico de falso negativo con `shap.waterfall_plot`. La predicción final (f(x) = 0.39) quedó por debajo del umbral de decisión (0.5), aunque la mayoría de variables relevantes (Diferencia_Temp +0.15, Rotational speed +0.11, Torque +0.11) empujaban correctamente hacia la clase falla. Esto indica que este error específico es un **caso límite de calibración** —el modelo identificó la dirección correcta pero no alcanzó el umbral— y no un error de interpretación equivocada del patrón.

## Verificación de data leakage

Se confirmó explícitamente (`X_train.columns`) que las features usadas excluyen identificadores y columnas de tipo de falla específico, descartando que los buenos resultados de Random Forest se deban a fuga de información.

## API — Endpoint de predicción

**POST** `/predict`

Recibe datos crudos de sensor y regresa la probabilidad de falla.

**Body de ejemplo:**
```json
{
  "air_temperature": 298.8,
  "process_temperature": 309,
  "rotational_speed": 1523,
  "torque": 38.6,
  "tool_wear": 177,
  "type": "L"
}
```

**Respuesta:**
```json
{
  "probabilidad_fallo": 0.55
}
```

El endpoint aplica internamente el mismo pipeline de feature engineering usado en entrenamiento (`features.py`): cálculo de potencia, diferencia de temperatura, y encoding ordinal de `Type`.

**Nota:** una probabilidad de 0.0 o 1.0 exactos no es un error — es el comportamiento esperado del modelo cuando los valores de sensor caen claramente en zonas sin ambigüedad (ver sección de EDA). Para probar el endpoint con casos de riesgo real, usar valores de torque en los extremos (5-15 o 50-70 Nm) o tool wear alto.

## Cómo correr este proyecto

### Opción 1: Con Docker (recomendado)

```bash
docker build -t machine-failure-api .
docker run -p 8000:8000 machine-failure-api
```

Una vez corriendo, abre `http://127.0.0.1:8000/docs` para probar el endpoint `/predict` de forma interactiva.

### Opción 2: Entorno local

```bash
pip install -r requirements.txt
python machine-failure-prediction.py   # entrena y guarda el modelo
uvicorn api:app --reload                # levanta la API
```

## Próximos pasos

- [ ] Diagnóstico de por qué GridSearchCV subóptimo respecto al modelo default

## Tecnologías

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · SHAP · FastAPI · Docker

## Estructura del proyecto

```
machine-failure-prediction/
├── ai4i2020.csv
├── machine-failure-prediction.py   # EDA, entrenamiento, evaluación, SHAP
├── features.py                      # feature engineering reutilizable
├── api.py                           # endpoint FastAPI
├── RandomForestMachineFailure.pkl  # modelo entrenado
├── Dockerfile
├── requirements.txt
└── README.md
```