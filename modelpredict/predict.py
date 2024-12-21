import pandas as pd
import pickle
import os
from datetime import datetime

with open('artifacts/pipeline.pkl','rb') as f:
    loans_model = pickle.load(f)

train_data = pd.read_csv('data/processed/feature_for_models.csv')
test_data = pd.read_csv('data/processed/test_dataset.csv')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV #para hiper-parámetros

# metricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

x_features = train_data.drop(labels=['SegmentoComercial'], axis = 1)
y_target = train_data['SegmentoComercial']

x_features_test = test_data.drop(labels=['SegmentoComercial'], axis = 1)
y_target_test = test_data['SegmentoComercial']

import mlflow

#configuración de conexión
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('Modelo Prediccion Clasificacion Prestamos')

predictions_folder = 'data/predictions'
timestamp = datetime.now().strftime("%Y%m%d%H")

with mlflow.start_run():
    # Configuración 1 Random Forest
    rf_model_1 = RandomForestClassifier(
        n_estimators=100,           # Número de árboles en el bosque
        max_depth=10,               # Profundidad máxima de los árboles
        min_samples_split=5,        # Número mínimo de muestras requeridas para dividir un nodo
        min_samples_leaf=3,         # Número mínimo de muestras en una hoja
        max_features='sqrt',        # Número de características a considerar al buscar la mejor división
        random_state=42             # Fijar la semilla para reproducibilidad
    )

    # Entrenar el modelo
    rf_model_1.fit(x_features, y_target)

    # Predecir
    rf_preds_1 = rf_model_1.predict(x_features_test)

    # Calcular la precisión
    rf_acc_1 = accuracy_score(y_target_test, rf_preds_1)

    # Registrar hiperparámetros del modelo 1
    mlflow.log_params({
        "rf1estimators": 100,
        "rf1_maxdepth": 10,
        "rf1_min_samp": 5,
        "rf1min_samleaf": 3,
        "rf1max_feat": "sqrt",
        "rf1randomstate": 42
    })

    #Guardar predicciones en CSV
    rf_preds_1_df = pd.DataFrame({'Predicciones': rf_preds_1})
    file_path_1 = os.path.join(predictions_folder, f"rfpred1_{timestamp}.csv")
    rf_preds_1_df.to_csv(file_path_1, index=False)
    mlflow.log_artifact(file_path_1)

    # Mostrar la precisión
    print(f"Accuracy of Random Forest Model 1: {rf_acc_1:.4f}")

    # Configuración 2 Random Forest
    rf_model_2 = RandomForestClassifier(
        n_estimators=200,           # Número de árboles en el bosque
        max_depth=5,               # Profundidad máxima de los árboles
        min_samples_split=8,        # Número mínimo de muestras requeridas para dividir un nodo
        min_samples_leaf=3,         # Número mínimo de muestras en una hoja
        max_features='sqrt',        # Número de características a considerar al buscar la mejor división
        random_state=2024             # Fijar la semilla para reproducibilidad
    )

    # Entrenar el modelo
    rf_model_2.fit(x_features, y_target)

    # Predecir
    rf_preds_2 = rf_model_2.predict(x_features_test)

    # Calcular la precisión
    rf_acc_2 = accuracy_score(y_target_test, rf_preds_2)

    # Registrar hiperparámetros del modelo 2
    mlflow.log_params({
        "rf2estimators": 200,
        "rf2_maxdepth": 5,
        "rf2_min_samp": 8,
        "rf2min_samleaf": 3,
        "rf2max_feat": "sqrt",
        "rf2randomstate": 2024
    })

    #Guardar predicciones en CSV
    rf_preds_2_df = pd.DataFrame({'Predicciones': rf_preds_2})
    file_path_2 = os.path.join(predictions_folder, f"rfpred2_{timestamp}.csv")
    rf_preds_2_df.to_csv(file_path_2, index=False)
    mlflow.log_artifact(file_path_2)

    # Mostrar la precisión
    print(f"Accuracy of Random Forest Model 2: {rf_acc_2:.4f}")


    # Configuración 3 Random Forest
    rf_model_3 = RandomForestClassifier(
        n_estimators=300,           # Número de árboles en el bosque
        max_depth=10,               # Profundidad máxima de los árboles
        min_samples_split=10,        # Número mínimo de muestras requeridas para dividir un nodo
        min_samples_leaf=3,         # Número mínimo de muestras en una hoja
        max_features='sqrt',        # Número de características a considerar al buscar la mejor división
        random_state=40             # Fijar la semilla para reproducibilidad
    )

    # Entrenar el modelo
    rf_model_3.fit(x_features, y_target)

    # Predecir
    rf_preds_3 = rf_model_3.predict(x_features_test)

    # Calcular la precisión
    rf_acc_3 = accuracy_score(y_target_test, rf_preds_3)

    # Registrar hiperparámetros del modelo 3
    mlflow.log_params({
        "rf3estimators": 300,
        "rf3_maxdepth": 10,
        "rf3_min_samp": 10,
        "rf3min_samleaf": 3,
        "rf3max_feat": "sqrt",
        "rf3randomstate": 40
    })

    #Guardar predicciones en CSV
    rf_preds_3_df = pd.DataFrame({'Predicciones': rf_preds_3})
    file_path_3 = os.path.join(predictions_folder, f"rfpred3_{timestamp}.csv")
    rf_preds_3_df.to_csv(file_path_3, index=False)
    mlflow.log_artifact(file_path_3)

    # Mostrar la precisión
    print(f"Accuracy of Random Forest Model 3: {rf_acc_3:.4f}")

    # registro de métrica de accuracy
    mlflow.log_metric("Acc RF1", rf_acc_1)
    mlflow.log_metric("Acc RF2", rf_acc_2)
    mlflow.log_metric("Acc RF3", rf_acc_3)

    #registro modelo de random forest
    mlflow.sklearn.log_model(rf_model_1, "RF1")
    mlflow.sklearn.log_model(rf_model_2, "RF2")
    mlflow.sklearn.log_model(rf_model_3, "RF3")

    mlflow.end_run()

with mlflow.start_run():
    # Configuración 1 Regresión Logistica
    lr_model_1 = LogisticRegression(
        penalty='l2',              # Regularización L2
        C=1.0,                     # Inverso de la fuerza de regularización
        solver='liblinear',        # Solución para pequeños datasets
        max_iter=100,              # Iteraciones máximas
        random_state=42
    )
    lr_model_1.fit(x_features, y_target)
    lr_preds_1 = lr_model_1.predict(x_features_test)
    lr_acc_1 = accuracy_score(y_target_test, lr_preds_1)

    # Registrar hiperparámetros del modelo 1
    mlflow.log_params({
        "lr1penalty": "l2",
        "lr1C": 1.0,
        "lr1solver": "liblinear",
        "lr1max_iter": 100,
        "lr1randomstate": 42
    })

    #Guardar predicciones en CSV
    lr_preds_1_df = pd.DataFrame({'Predicciones': lr_preds_1})
    file_path_1 = os.path.join(predictions_folder, f"lrpred1_{timestamp}.csv")
    lr_preds_1_df.to_csv(file_path_1, index=False)
    mlflow.log_artifact(file_path_1)

    # Mostrar la precisión
    print(f"Accuracy of Logistic Regression 1: {lr_acc_1:.4f}")

    # Configuración 2
    lr_model_2 = LogisticRegression(
        penalty='l1',              # Regularización L1
        C=0.5,                     # Fuerza de regularización más alta
        solver='saga',             # Compatible con L1
        max_iter=200,
        random_state=42
    )
    lr_model_2.fit(x_features, y_target)
    lr_preds_2 = lr_model_2.predict(x_features_test)
    lr_acc_2 = accuracy_score(y_target_test, lr_preds_2)

    # Registrar hiperparámetros del modelo 2
    mlflow.log_params({
        "lr2penalty": "l1",
        "lr2C": 0.5,
        "lr2solver": "saga",
        "lr2max_iter": 200,
        "lr2randomstate": 42
    })

    #Guardar predicciones en CSV
    lr_preds_2_df = pd.DataFrame({'Predicciones': lr_preds_2})
    file_path_2 = os.path.join(predictions_folder, f"lrpred2_{timestamp}.csv")
    lr_preds_2_df.to_csv(file_path_2, index=False)
    mlflow.log_artifact(file_path_2)

    # Mostrar la precisión
    print(f"Accuracy of Logistic Regression 2: {lr_acc_2:.4f}")

    # Configuración 3
    lr_model_3 = LogisticRegression(
        penalty='elasticnet',      # ElasticNet combina L1 y L2
        C=0.1,
        solver='saga',
        l1_ratio=0.5,              # Proporción de L1 en ElasticNet
        max_iter=300,
        random_state=42
    )
    lr_model_3.fit(x_features, y_target)
    lr_preds_3 = lr_model_3.predict(x_features_test)
    lr_acc_3 = accuracy_score(y_target_test, lr_preds_3)

    # Registrar hiperparámetros del modelo 3
    mlflow.log_params({
        "lr3penalty": "elasticnet",
        "lr3C": 0.1,
        "lr3solver": "saga",
        "lr3l1_ratio": 0.5,
        "lr3max_iter": 300,
        "lr3randomstate": 42
    })

    #Guardar predicciones en CSV
    lr_preds_3_df = pd.DataFrame({'Predicciones': lr_preds_3})
    file_path_3 = os.path.join(predictions_folder, f"lrpred3_{timestamp}.csv")
    lr_preds_3_df.to_csv(file_path_3, index=False)
    mlflow.log_artifact(file_path_3)

    # Mostrar la precisión
    print(f"Accuracy of Logistic Regression 3: {lr_acc_3:.4f}")

    # registro de métrica de accuracy
    mlflow.log_metric("Acc LR1", lr_acc_1)
    mlflow.log_metric("Acc LR2", lr_acc_2)
    mlflow.log_metric("Acc LR3", lr_acc_3)

    #registro modelo de random forest
    mlflow.sklearn.log_model(lr_model_1, "LR1")
    mlflow.sklearn.log_model(lr_model_2, "LR2")
    mlflow.sklearn.log_model(lr_model_3, "LR3")

    mlflow.end_run()

with mlflow.start_run():
    # Configuración 1 SVC
    svm_model_1 = SVC(
        kernel='linear',           # Kernel lineal
        C=1.0,                     # Regularización estándar
        probability=True,          # Probabilidades habilitadas
        random_state=42
    )
    svm_model_1.fit(x_features, y_target)
    svm_preds_1 = svm_model_1.predict(x_features_test)
    svm_acc_1 = accuracy_score(y_target_test, svm_preds_1)

    # Registrar hiperparámetros del modelo 1
    mlflow.log_params({
        "svm1kernel": "linear",
        "svm1C": 1.0,
        "svm1prob": "True",
        "svm1randomstate": 42
    })

    #Guardar predicciones en CSV
    svm_preds_1_df = pd.DataFrame({'Predicciones': svm_preds_1})
    file_path_1 = os.path.join(predictions_folder, f"svpred1_{timestamp}.csv")
    svm_preds_1_df.to_csv(file_path_1, index=False)
    mlflow.log_artifact(file_path_1)

    # Mostrar la precisión
    print(f"Accuracy of SVM 1: {svm_acc_1:.4f}")

    # Configuración 2 SVC
    svm_model_2 = SVC(
        kernel='rbf',              # Kernel Gaussiano
        C=0.5,
        gamma='scale',             # Parámetro gamma automático
        random_state=42
    )
    svm_model_2.fit(x_features, y_target)
    svm_preds_2 = svm_model_2.predict(x_features_test)
    svm_acc_2 = accuracy_score(y_target_test, svm_preds_2)

    # Registrar hiperparámetros del modelo 2
    mlflow.log_params({
        "svm2kernel": "rbf",
        "svm2C": 0.5,
        "svm2gamma": "scale",
        "svm2randomstate": 42
    })

    #Guardar predicciones en CSV
    svm_preds_2_df = pd.DataFrame({'Predicciones': svm_preds_2})
    file_path_2 = os.path.join(predictions_folder, f"svpred2_{timestamp}.csv")
    svm_preds_2_df.to_csv(file_path_2, index=False)
    mlflow.log_artifact(file_path_2)

    # Mostrar la precisión
    print(f"Accuracy of SVM 2: {svm_acc_2:.4f}")

    # Configuración 3 SVC
    svm_model_3 = SVC(
        kernel='poly',             # Kernel polinomial
        degree=3,                  # Grado del polinomio
        C=1.5,
        probability=True,
        random_state=42
    )
    svm_model_3.fit(x_features, y_target)
    svm_preds_3 = svm_model_3.predict(x_features_test)
    svm_acc_3 = accuracy_score(y_target_test, svm_preds_3)

    # Registrar hiperparámetros del modelo 3
    mlflow.log_params({
        "svm3kernel": "poly",
        "svm3degree": 3,
        "svm3C": 1.5,
        "svm3prob": "True",
        "svm3randomstate": 42
    })

    #Guardar predicciones en CSV
    svm_preds_3_df = pd.DataFrame({'Predicciones': svm_preds_3})
    file_path_3 = os.path.join(predictions_folder, f"svpred3_{timestamp}.csv")
    svm_preds_3_df.to_csv(file_path_3, index=False)
    mlflow.log_artifact(file_path_3)

    # Mostrar la precisión
    print(f"Accuracy of SVM 3: {svm_acc_3:.4f}")

    # registro de métrica de accuracy
    mlflow.log_metric("Acc SVM1", svm_acc_1)
    mlflow.log_metric("Acc SVM2", svm_acc_2)
    mlflow.log_metric("Acc SVM3", svm_acc_3)

    #registro modelo de random forest
    mlflow.sklearn.log_model(svm_model_1, "SVM1")
    mlflow.sklearn.log_model(svm_model_2, "SVM2")
    mlflow.sklearn.log_model(svm_model_3, "SVM3")

    mlflow.end_run()

with mlflow.start_run():
    # Configuración 1 AdaBoost Classifier
    adaboost_model_1 = AdaBoostClassifier(
        n_estimators=50,           # Número de estimadores
        learning_rate=1.0,         # Tasa de aprendizaje estándar
        random_state=42
    )
    adaboost_model_1.fit(x_features, y_target)
    adaboost_preds_1 = adaboost_model_1.predict(x_features_test)
    adaboost_acc_1 = accuracy_score(y_target_test, adaboost_preds_1)

    # Registrar hiperparámetros del modelo 1
    mlflow.log_params({
        "ada1estimators": 50,
        "ada1learnrate": 1.0,
        "ada1randomstate": 42
    })

    #Guardar predicciones en CSV
    adaboost_preds_1_df = pd.DataFrame({'Predicciones': adaboost_preds_1})
    file_path_1 = os.path.join(predictions_folder, f"adpred1_{timestamp}.csv")
    adaboost_preds_1_df.to_csv(file_path_1, index=False)
    mlflow.log_artifact(file_path_1)

    # Mostrar la precisión
    print(f"Accuracy of AdaBoost 1: {adaboost_acc_1:.4f}")

    # Configuración 2 AdaBoost Classifier
    adaboost_model_2 = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.5,         # Tasa de aprendizaje más baja
        random_state=42
    )
    adaboost_model_2.fit(x_features, y_target)
    adaboost_preds_2 = adaboost_model_2.predict(x_features_test)
    adaboost_acc_2 = accuracy_score(y_target_test, adaboost_preds_2)

    # Registrar hiperparámetros del modelo 2
    mlflow.log_params({
        "ada2estimators": 100,
        "ada2learnrate": 0.5,
        "ada2randomstate": 42
    })

    #Guardar predicciones en CSV
    adaboost_preds_2_df = pd.DataFrame({'Predicciones': adaboost_preds_2})
    file_path_2 = os.path.join(predictions_folder, f"adpred2_{timestamp}.csv")
    adaboost_preds_2_df.to_csv(file_path_2, index=False)
    mlflow.log_artifact(file_path_2)

    # Mostrar la precisión
    print(f"Accuracy of AdaBoost 2: {adaboost_acc_2:.4f}")

    # Configuración 3 AdaBoost Classifier
    adaboost_model_3 = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.1,         # Tasa de aprendizaje muy baja
        random_state=42
    )
    adaboost_model_3.fit(x_features, y_target)
    adaboost_preds_3 = adaboost_model_3.predict(x_features_test)
    adaboost_acc_3 = accuracy_score(y_target_test, adaboost_preds_3)

    # Registrar hiperparámetros del modelo 3
    mlflow.log_params({
        "ada3estimators": 200,
        "ada3learnrate": 0.1,
        "ada3randomstate": 42
    })

    #Guardar predicciones en CSV
    adaboost_preds_3_df = pd.DataFrame({'Predicciones': adaboost_preds_3})
    file_path_3 = os.path.join(predictions_folder, f"adpred3_{timestamp}.csv")
    adaboost_preds_3_df.to_csv(file_path_3, index=False)
    mlflow.log_artifact(file_path_3)

    # Mostrar la precisión
    print(f"Accuracy of AdaBoost 3: {adaboost_acc_3:.4f}")

    # registro de métrica de accuracy
    mlflow.log_metric("Acc Ada1", adaboost_acc_1)
    mlflow.log_metric("Acc Ada2", adaboost_acc_2)
    mlflow.log_metric("Acc Ada3", adaboost_acc_3)

    #registro modelo de random forest
    mlflow.sklearn.log_model(adaboost_model_1, "Ada1")
    mlflow.sklearn.log_model(adaboost_model_2, "Ada2")
    mlflow.sklearn.log_model(adaboost_model_3, "Ada3")

    mlflow.end_run()

with mlflow.start_run():
    # Configuración 1 XGBoost
    xgb_model_1 = xgb.XGBClassifier(
        n_estimators=100,          # Número de árboles
        max_depth=6,               # Profundidad máxima de los árboles
        learning_rate=0.3,         # Tasa de aprendizaje estándar
        random_state=42,
        use_label_encoder=False
    )
    xgb_model_1.fit(x_features, y_target)
    xgb_preds_1 = xgb_model_1.predict(x_features_test)
    xgb_acc_1 = accuracy_score(y_target_test, xgb_preds_1)

    # Registrar hiperparámetros del modelo 1
    mlflow.log_params({
        "XGB1estimators": 100,
        "XGB1maxdepth": 6,
        "XGB1learnrate": 0.3,
        "XGB1randomstate": 42
    })

    #Guardar predicciones en CSV
    xgb_preds_1_df = pd.DataFrame({'Predicciones': xgb_preds_1})
    file_path_1 = os.path.join(predictions_folder, f"xgpred1_{timestamp}.csv")
    xgb_preds_1_df.to_csv(file_path_1, index=False)
    mlflow.log_artifact(file_path_1)

    # Mostrar la precisión
    print(f"Accuracy of XGBoost 1: {xgb_acc_1:.4f}")

    # Configuración 2 XGBoost
    xgb_model_2 = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,             # Submuestreo de datos
        colsample_bytree=0.8,      # Submuestreo de características
        random_state=42,
        use_label_encoder=False
    )
    xgb_model_2.fit(x_features, y_target)
    xgb_preds_2 = xgb_model_2.predict(x_features_test)
    xgb_acc_2 = accuracy_score(y_target_test, xgb_preds_2)

    # Registrar hiperparámetros del modelo 2
    mlflow.log_params({
        "XGB2estimators": 200,
        "XGB2maxdepth": 4,
        "XGB2learnrate": 0.1,
        "XGB2subsample": 0.8,
        "XGB2colbytree": 0.8,
        "XGB2randomstate": 42
    })

    #Guardar predicciones en CSV
    xgb_preds_2_df = pd.DataFrame({'Predicciones': xgb_preds_2})
    file_path_2 = os.path.join(predictions_folder, f"xgpred2_{timestamp}.csv")
    xgb_preds_2_df.to_csv(file_path_2, index=False)
    mlflow.log_artifact(file_path_2)

    # Mostrar la precisión
    print(f"Accuracy of XGBoost 2: {xgb_acc_2:.4f}")

    # Configuración 3 XGBoost
    xgb_model_3 = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.6,
        colsample_bytree=0.6,
        gamma=1.0,                 # Regularización
        random_state=42,
        use_label_encoder=False
    )
    xgb_model_3.fit(x_features, y_target)
    xgb_preds_3 = xgb_model_3.predict(x_features_test)
    xgb_acc_3 = accuracy_score(y_target_test, xgb_preds_3)

    # Registrar hiperparámetros del modelo 3
    mlflow.log_params({
        "XGB3estimators": 300,
        "XGB3maxdepth": 8,
        "XGB3learnrate": 0.05,
        "XGB3subsample": 0.6,
        "XGB3colbytree": 0.6,
        "XGB3gamma": 1.0,
        "XGB3randomstate": 42
    })

    #Guardar predicciones en CSV
    xgb_preds_3_df = pd.DataFrame({'Predicciones': xgb_preds_3})
    file_path_3 = os.path.join(predictions_folder, f"xgpred3_{timestamp}.csv")
    xgb_preds_3_df.to_csv(file_path_3, index=False)
    mlflow.log_artifact(file_path_3)

    # Mostrar la precisión
    print(f"Accuracy of XGBoost 3: {xgb_acc_3:.4f}")

    # registro de métrica de accuracy
    mlflow.log_metric("Acc XGB1", xgb_acc_1)
    mlflow.log_metric("Acc XGB2", xgb_acc_2)
    mlflow.log_metric("Acc XGB3", xgb_acc_3)

    #registro modelo de random forest
    mlflow.sklearn.log_model(xgb_model_1, "XGB")
    mlflow.sklearn.log_model(xgb_model_2, "XGB")
    mlflow.sklearn.log_model(xgb_model_3, "XGB")

    mlflow.end_run()
