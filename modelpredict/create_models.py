"""
Este módulo contiene la implementación de varios algoritmos de clasificación
para evaluar el rendimiento de distintos modelos en un conjunto de datos 
relacionado con la rentabilidad de una cartera crediticia.

El script incluye:
- Modelos de regresión logística y árboles de decisión.
- Evaluación de hiperparámetros usando GridSearchCV.
- Evaluación del rendimiento utilizando métricas como la precisión y el recall.
"""
# %% [markdown]
# ##### **Universidad Galileo**
# ##### **Postgrado en Análisis y Predicción de Datos** - *Cuarto Trimestre 2024*
# ##### **Product Development**
# ##### **Alumno:** Elder Cruz. **Carnet:** 23004456
# ##### **Alumno:** Walter Reyes. **Carnet:** 23004450
# ##### **Alumno:** Dayana Gamboa. **Carnet:** 23001913
# ##### **Alumno:** Rodrigo Cano. **Carnet:** 23001916
# ##### **Proyecto final:** Create models
# %% [markdown]
# # 1. Importación de librerías

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV #para hiper-parámetros

# metricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# gestion train-test
from sklearn.model_selection import train_test_split

# transformaciones
from sklearn.preprocessing import MinMaxScaler

# %% [markdown]
# # 2. Carga de datos

# %%
DATASET_PATH ='../data/processed/df_loans_FS.csv'

# %%
df_Loans = pd.read_csv(DATASET_PATH, delimiter = ",")
df_Loans.head()

# %% [markdown]
# # 3. EDA - Exploratory Analysis

# %%
df = df_Loans.drop(columns=['Desembolso','Vencimiento','FechaReporte'])

df.head()

# %%
df['SegmentoComercial'].value_counts(normalize=True)

#Recordando los segmentos:
#{0: '1_INMOBILIARIA', 1: '2_NEGOCIO', 2: '3_COMERCIAL', 3: '4_RETAIL', 4: '5_INTERCOMPANY'}

# %% [markdown]
# # 4. Balanceo de clases en target

# %%
dataPositiva = df[df['SegmentoComercial'] == 0]
# %%
dataNegativa = df[df['SegmentoComercial'] == 3]
# %%
n = 2*dataPositiva.shape[0]
dataNegativa = dataNegativa.sample(n, random_state=2024, replace = False)
# %%
completeData = pd.concat([dataPositiva,dataNegativa])
completeData = completeData.sample(len(completeData), replace = False, random_state = 2025)

# %%
completeData['SegmentoComercial'].value_counts(normalize=True)

# %%
def encode_target(value):
    """
    Invierte el valor de entrada, retornando 1 si el valor es 0, y 0 si el valor es 1.
    
    Parameters:
    value (int): El valor que se desea invertir, debe ser 0 o 1.

    Returns:
    int: El valor invertido (1 si el valor original era 0, 0 si el valor original era 1).
    """
    if value == 0:
        return 1
    return 0
# %%
completeData['SegmentoComercial'] = completeData['SegmentoComercial'].map(encode_target)
completeData.head()

# %% [markdown]
# # 5. Construcción de Modelos
# %%
completeData.head()

# %%
# seleccionamos el target y features
X = completeData.drop('SegmentoComercial', axis = 1)
y = completeData['SegmentoComercial']

# %%
# split para train y test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=2025,shuffle=True)

# %%
# creamos el scaler
scaler = MinMaxScaler()

scaler.fit(X_train) #calculamos el scaler

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## 5.3 Regresión logística

# %%
logit = LogisticRegression()
logit.fit(X_train_scaled, y_train) #Entrenamiento del modelo

logit_predicts = logit.predict(X_test_scaled) #predicciones
roc_auc = roc_auc_score(logit_predicts, y_test)

print("Roc_Auc Score:", roc_auc)
print("Accuracy:", accuracy_score(y_test, logit_predicts))
# %%
np.array(y_test)

# %%
intercept = logit.intercept_
print(f'Intercepto: {intercept}')

coefficients = logit.coef_
print(f'Coeficientes: {coefficients}')

# %%
# Obtener las probabilidades de la clase positiva (1)
logit_probabilities = logit.predict_proba(X_test_scaled)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, logit_probabilities)
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# %% [markdown]
# ### 5.3.1 Optimización con hyperparámetros

# %%
# Crear el clasificador
lr_classifier = LogisticRegression(max_iter=1000)

# Definir el grid de hiperparámetros
hyper_params_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2", "elasticnet", "none"],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "l1_ratio": [0.1, 0.5, 0.7]  # Solo aplicable cuando penalty es 'elasticnet'
}

# Configuración para la optimización de hiperparámetros
# Asegurarse de manejar combinaciones inválidas
param_grid = [
    {"solver": ["newton-cg", "lbfgs", "sag"], "penalty": ["l2"], "C": [0.01, 0.1, 1, 10, 100]},
    {"solver": ["liblinear"], "penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10, 100]},
    {"solver": ["saga"], "penalty": ["l1", "l2", "elasticnet"], "C": [0.01, 0.1, 1, 10, 100], "l1_ratio": [0.1, 0.5, 0.7]},
    {"solver": ["saga"], "penalty": ["none"]}
]

# Configuración para la optimización de hiperparámetros
lr_hyp_opt = GridSearchCV(estimator=lr_classifier, param_grid=param_grid, cv=10, scoring="roc_auc")

# Entrenar el modelo
lr_hyp_opt.fit(X_train_scaled, y_train)

# Mostrar los mejores hiperparámetros
print("Mejores hiperparámetros:", lr_hyp_opt.best_params_)

# %%
# score del mejor modelo

# %%
ls_results = pd.DataFrame(lr_hyp_opt.cv_results_).sort_values("rank_test_score", ascending=True)[['rank_test_score', 'params', 'mean_test_score', 'std_test_score']]
#nos mostrará un score de los mejores resultados

# %% [markdown]
# ## 5.5 Árboles de Decisión

# %%
DTC_Classifier = DecisionTreeClassifier()
DTC_Classifier.fit(X_train_scaled, y_train) #Entrenamiento del modelo

DTC_predicts = DTC_Classifier.predict(X_test_scaled) #predicciones
roc_auc = roc_auc_score(DTC_predicts, y_test)

print("Roc_Auc Score:", roc_auc)
print("Accuracy:", accuracy_score(y_test, DTC_predicts))

# %%
np.array(y_test)

# %% [markdown]
# ### 5.5.1 Árboles de Decisión Optimización de Hyper-Parámetros

# %%
# Crear el clasificador Decision Tree
dt_classifier = DecisionTreeClassifier()

# Definir el grid de hiperparámetros
hyper_params_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "max_features": [None, "sqrt", "log2"]
}

# Configuración para la optimización de hiperparámetros
dt_hyp_opt = GridSearchCV(estimator=dt_classifier, param_grid=hyper_params_grid, cv=10, scoring="roc_auc")

# Entrenar el modelo
dt_hyp_opt.fit(X_train_scaled, y_train)

# Mostrar los mejores hiperparámetros
print("Mejores hiperparámetros:", dt_hyp_opt.best_params_)

# %%
# score del mejor modelo

# %%
dt_results = pd.DataFrame(dt_hyp_opt.cv_results_).sort_values("rank_test_score", ascending=True)[['rank_test_score', 'params', 'mean_test_score', 'std_test_score']]
#nos mostrará un score de los mejores resultados

# %% [markdown]
# # 6. Matriz de resultados de modelos

# %%
# Convertir los resultados a DataFrames
df_lr = pd.DataFrame(lr_hyp_opt.cv_results_).sort_values("rank_test_score", ascending=True)[['rank_test_score', 'params', 'mean_test_score', 'std_test_score']]
df_dt = pd.DataFrame(dt_hyp_opt.cv_results_).sort_values("rank_test_score", ascending=True)[['rank_test_score', 'params', 'mean_test_score', 'std_test_score']]

# Añadir una columna para identificar el modelo
df_lr['model'] = 'Logistic Regression'
df_dt['model'] = 'Decision Tree'

# Concatenar los resultados en un solo DataFrame
df_results = pd.concat([df_lr, df_dt], ignore_index=True)

df_results_sorted = df_results.sort_values("mean_test_score", ascending = False)
# %%
# Guardar los mejores modelos en un archivo CSV
df_results_sorted.to_csv('../data/processed/df_results_sorted.csv', index=False)

# O guardar los mejores modelos en un archivo Excel
df_results_sorted.to_excel('../data/processed/df_results_sorted.xlsx', index=False)
