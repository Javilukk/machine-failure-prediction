from turtle import speed
import joblib
import pandas as pd
import seaborn 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import shap
from features import crear_feature_potencia, crear_feature_diferencia_temp, codificar_type

dataset = pd.read_csv('C:\\Users\\marte\\OneDrive\\Documentos\\Escritorio\\Codigos\\CodigosPython\\machine-failure-prediction\\ai4i2020.csv')
pd.set_option('display.max_columns', None)


def generar_histograma():
    seaborn.histplot(data=dataset, x='Torque [Nm]', hue='PWF', common_norm=False, stat="density", element="step")
    print(dataset.shape)
    print(dataset.info())
    print(dataset.describe())
    print(dataset['Machine failure'].value_counts(normalize=True))
    plt.show()
    plt.title('Histograma de fallas de la máquina')
    plt.xlabel('Fallas de la máquina')
    plt.ylabel('Densidad')
    print("Histograma de fallas de la máquina generado con Seaborn.")
dataset = crear_feature_potencia(dataset)
dataset = crear_feature_diferencia_temp(dataset)
dataset = codificar_type(dataset)


X_train, X_test, y_train, y_test = train_test_split(dataset[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Diferencia_Temp', 'Potencia', 'Type_encoded']], 
                                                    dataset['Machine failure'], test_size=0.3, stratify=dataset['Machine failure'], random_state=42)



def dummyclassifier(X_train, X_test, y_train, y_test):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    y_pred = dummy_clf.predict(X_test)
    print(classification_report(y_test, y_pred))



def logisticRegression(X_train, X_test, y_train, y_test):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print(classification_report(y_test, y_pred))


def RandomForest(X_train, X_test, y_train, y_test):

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42,)  
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    falsos_negativos = X_test[(y_test == 1) & (y_pred == 0)]
    verdaderos_positivos = X_test[(y_test == 1) & (y_pred == 1)]
    joblib.dump(rf_clf, 'RandomForestMachineFailure.pkl')
    return rf_clf, falsos_negativos, verdaderos_positivos

modelo_entrenado, falsos_negativos, verdaderos_positivos = RandomForest(X_train, X_test, y_train, y_test)

def XgboostModel(X_train, X_test, y_train, y_test):

    X_train = X_train.rename(columns={
        'Torque [Nm]': 'Torque_Nm',
        'Rotational speed [rpm]': 'Rotational_speed_rpm',
        'Tool wear [min]': 'Tool_wear_min',
        'Air temperature [K]': 'Air_temperature_K',
        'Process temperature [K]': 'Process_temperature_K'
    })

    X_test = X_test.rename(columns={
        'Torque [Nm]': 'Torque_Nm',
        'Rotational speed [rpm]': 'Rotational_speed_rpm',
        'Tool wear [min]': 'Tool_wear_min',
        'Air temperature [K]': 'Air_temperature_K',
        'Process temperature [K]': 'Process_temperature_K'
    })
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss')
    scale_pos_weight = (len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
    xgb_clf.set_params(scale_pos_weight=scale_pos_weight)
    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)
    print(classification_report(y_test, y_pred))


def XgboostModelNoScale(X_train, X_test, y_train, y_test):

    X_train = X_train.rename(columns={
        'Torque [Nm]': 'Torque_Nm',
        'Rotational speed [rpm]': 'Rotational_speed_rpm',
        'Tool wear [min]': 'Tool_wear_min',
        'Air temperature [K]': 'Air_temperature_K',
        'Process temperature [K]': 'Process_temperature_K'
    })

    X_test = X_test.rename(columns={
        'Torque [Nm]': 'Torque_Nm',
        'Rotational speed [rpm]': 'Rotational_speed_rpm',
        'Tool wear [min]': 'Tool_wear_min',
        'Air temperature [K]': 'Air_temperature_K',
        'Process temperature [K]': 'Process_temperature_K'
    })

    xgb_clf = xgb.XGBClassifier(eval_metric='logloss')
    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)
    print(classification_report(y_test, y_pred))


def GridSearchCv (X_train, X_test, y_train, y_test):
    randomforest = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }
    grid_search = GridSearchCV(estimator=randomforest, param_grid=param_grid, scoring='f1', cv=5, verbose=2)
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(classification_report(y_test, y_pred))

    print(grid_search.best_score_)


def shapTreeGlobal(modelo, xtest,):
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(xtest)
    shap_values_falla = shap_values[:, :, 1]
    
    shap.summary_plot(shap_values_falla, xtest, plot_type="bar")
    shap.summary_plot(shap_values_falla, xtest)

    shap.dependence_plot("Torque [Nm]", shap_values_falla, xtest)
    shap.dependence_plot("Rotational speed [rpm]", shap_values_falla, xtest)
    shap.dependence_plot("Tool wear [min]", shap_values_falla, xtest)
    shap.dependence_plot("Air temperature [K]", shap_values_falla, xtest)
    shap.dependence_plot("Process temperature [K]", shap_values_falla, xtest)
    shap.dependence_plot("Diferencia_Temp", shap_values_falla, xtest)
    shap.dependence_plot("Potencia", shap_values_falla, xtest)
    shap.dependence_plot("Type_encoded", shap_values_falla, xtest)

def shap_caso_individual(modelo, xtest, falsos_negativos):
    explainer = shap.TreeExplainer(modelo)
    indice_fila = falsos_negativos.index[1]
    posicion = xtest.index.get_loc(indice_fila)

    shap_explanation = explainer(xtest)

    shap.waterfall_plot(shap_explanation[posicion, :, 1])

fila_cualquiera = X_test.iloc[[0]]
print(fila_cualquiera)
print(modelo_entrenado.predict_proba(verdaderos_positivos.iloc[[0, 1, 2]]))