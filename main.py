import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Cargar datos
archivo_csv = "deportistas.csv"
df = pd.read_csv("D:/Usuarios/mgallegol04/Desktop/arbol_complex/clasificadores/deportistas.csv")

# Preprocesamiento
df = df.dropna(subset=["tipo_atleta"])
df["tipo_atleta"] = df["tipo_atleta"].map({"Fondista": 0, "Velocista": 1})

df = pd.get_dummies(df, columns=["sexo", "raza"], drop_first=True)
X = df.drop(columns=["tipo_atleta"])
y = df["tipo_atleta"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5)
modelo.fit(X_train_scaled, y_train)

# Guardar modelo correctamente
modelo_path = "modelo_atletas.pkl"
with open(modelo_path, "wb") as file:
    pickle.dump({"scaler": scaler, "modelo": modelo}, file)  # Guardamos en un diccionario

# Evaluación del modelo
y_pred = modelo.predict(X_test_scaled)
y_pred_proba = modelo.predict_proba(X_test_scaled)[:, 1]
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Guardar métricas en un archivo de texto
metrics_path = "metricas_modelo.txt"
with open(metrics_path, "w") as file:
    file.write("### Matriz de Confusión/n")
    file.write(str(conf_matrix) + "/n/n")
    file.write("### Reporte de Clasificación/n")
    file.write(class_report + "/n/n")
    file.write(f"### AUC-ROC: {roc_auc:.2f}/n")

print("Modelo y métricas guardadas correctamente.")
