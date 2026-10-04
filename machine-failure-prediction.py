from turtle import speed
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
from sklearn.datasets import make_classification

dataset = pd.read_csv('C:\\Users\\marte\\OneDrive\\Documentos\\Escritorio\\Codigos\\CodigosPython\\machine-failure-prediction\\ai4i2020.csv')

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

def crear_feature_potencia(df):
    Rads_per_sec = df['Rotational speed [rpm]'] * (2 * 3.141592653589793 / 60)
    df['Potencia'] = df['Torque [Nm]'] * Rads_per_sec
    return df

def crear_feature_diferencia_temp(df):
    df['Diferencia_Temp'] = df['Process temperature [K]'] - df['Air temperature [K]']
    return df

def codificar_type(df):
    dataframe= OrdinalEncoder(categories=[['L', 'M', 'H']]).fit_transform(df[['Type']])
    df['Type_encoded'] = dataframe
    return df


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
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    y_pred = rf_clf.predict(X_test)
    falsos_negativos = X_test[(y_test == 1) & (y_pred == 0)]
    print(falsos_negativos.describe())
    verdaderos_positivos = X_test[(y_test == 1) & (y_pred == 1)]
    print(verdaderos_positivos.describe())
    print(classification_report(y_test, y_pred))
pd.set_option('display.max_columns', None)

print(RandomForest(X_train, X_test, y_train, y_test))