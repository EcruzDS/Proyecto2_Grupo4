"""Este modulo contiene el código donde se realiza la selección del modelo ganador
"""
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

def configure_and_fit_pipeline():
    """Esta función crea la selección del modelo ganador
    """
    train_data = pd.read_csv('../data/processed/feature_for_models.csv')
    test_data = pd.read_csv('../data/processed/test_dataset.csv')

    with open('../artifacts/pipeline.pkl','rb') as f:
        loans_model = pickle.load(f)

    # #### Agregar lectura de target con el archivo de configuración
    x_features = train_data.drop(labels=['SegmentoComercial'], axis = 1)
    y_target = train_data['SegmentoComercial']

    x_features_test = test_data.drop(labels=['SegmentoComercial'], axis = 1)
    y_target_test = test_data['SegmentoComercial']

    # Configuración 1 Random Forest
    rf_model_1 = RandomForestClassifier(
        n_estimators=100,           # Número de árboles en el bosque
        max_depth=10,               # Profundidad máxima de los árboles
        min_samples_split=5,        # Número mínimo de muestras requeridas para dividir un nodo
        min_samples_leaf=3,         # Número mínimo de muestras en una hoja
        max_features='sqrt',        # Número de características a considerar
        random_state=42             # Fijar la semilla para reproducibilidad
    )

    # Entrenar el modelo
    rf_model_1.fit(x_features, y_target)

    # Predecir
    y_preds_1 = rf_model_1.predict(x_features_test)

    # Calcular la precisión
    rf_acc_1 = accuracy_score(y_target_test, y_preds_1)

    # Mostrar la precisión
    print(f"Accuracy of Random Forest Model 1: {rf_acc_1:.4f}")

    # Configuración 2 Random Forest
    rf_model_2 = RandomForestClassifier(
        n_estimators=200,           # Número de árboles en el bosque
        max_depth=5,               # Profundidad máxima de los árboles
        min_samples_split=8,        # Número mínimo de muestras requeridas para dividir un nodo
        min_samples_leaf=3,         # Número mínimo de muestras en una hoja
        max_features='sqrt',        # Número de características a considerar
        random_state=2024             # Fijar la semilla para reproducibilidad
    )

    # Entrenar el modelo
    rf_model_2.fit(x_features, y_target)

    # Predecir
    y_preds_2 = rf_model_2.predict(x_features_test)

    # Calcular la precisión
    rf_acc_2 = accuracy_score(y_target_test, y_preds_2)

    # Mostrar la precisión
    print(f"Accuracy of Random Forest Model 2: {rf_acc_2:.4f}")

    # Configuración 3 Random Forest
    rf_model_3 = RandomForestClassifier(
        n_estimators=300,           # Número de árboles en el bosque
        max_depth=10,               # Profundidad máxima de los árboles
        min_samples_split=10,        # Número mínimo de muestras requeridas para dividir un nodo
        min_samples_leaf=3,         # Número mínimo de muestras en una hoja
        max_features='sqrt',        # Número de características a considerar
        random_state=40             # Fijar la semilla para reproducibilidad
    )

    # Entrenar el modelo
    rf_model_3.fit(x_features, y_target)

    # Predecir
    y_preds_3 = rf_model_3.predict(x_features_test)

    # Calcular la precisión
    rf_acc_3 = accuracy_score(y_target_test, y_preds_3)

    # Mostrar la precisión
    print(f"Accuracy of Random Forest Model 3: {rf_acc_3:.4f}")

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

    # Mostrar la precisión
    print(f"Accuracy of Logistic Regression 3: {lr_acc_3:.4f}")

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

    # Mostrar la precisión
    print(f"Accuracy of SVM 3: {svm_acc_3:.4f}")

    # Configuración 1 AdaBoost Classifier
    adaboost_model_1 = AdaBoostClassifier(
        n_estimators=50,           # Número de estimadores
        learning_rate=1.0,         # Tasa de aprendizaje estándar
        random_state=42
    )
    adaboost_model_1.fit(x_features, y_target)
    adaboost_preds_1 = adaboost_model_1.predict(x_features_test)
    adaboost_acc_1 = accuracy_score(y_target_test, adaboost_preds_1)

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

    # Mostrar la precisión
    print(f"Accuracy of AdaBoost 3: {adaboost_acc_3:.4f}")

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

    # Mostrar la precisión
    print(f"Accuracy of XGBoost 3: {xgb_acc_3:.4f}")

    # Lista de todos los accuracies
    rf_acc = max(rf_acc_1, rf_acc_2, rf_acc_3)
    lr_acc = max(lr_acc_1, lr_acc_2, lr_acc_3)
    svm_acc = max(svm_acc_1, svm_acc_2, svm_acc_3)
    ada_acc = max(adaboost_acc_1, adaboost_acc_2, adaboost_acc_3)
    xgb_acc = max(xgb_acc_1, xgb_acc_2, xgb_acc_3)

    # Encontrar el mejor accuracy general
    modelos = [rf_acc, lr_acc, svm_acc, ada_acc, xgb_acc]
    mejormodelo = max(modelos)

    if mejormodelo == rf_acc:
        # Determinar la configuración específica de RandomForest
        if rf_acc == rf_acc_1:
            loans_model.steps.append(
                ('modelo_random_forest', RandomForestClassifier(
                    n_estimators=100, max_depth=6, random_state=42))
            )
        elif rf_acc == rf_acc_2:
            loans_model.steps.append(
                ('modelo_random_forest', RandomForestClassifier(
                    n_estimators=200, max_depth=8, random_state=42))
            )
        else:
            loans_model.steps.append(
                ('modelo_random_forest', RandomForestClassifier(
                    n_estimators=300, max_depth=10, random_state=42))
            )

    elif mejormodelo == lr_acc:
        # Determinar la configuración específica de LogisticRegression
        if lr_acc == lr_acc_1:
            loans_model.steps.append(
                ('modelo_logistic_regression', LogisticRegression(
                    C=1.0, penalty='l2', solver='lbfgs', random_state=42))
            )
        elif lr_acc == lr_acc_2:
            loans_model.steps.append(
                ('modelo_logistic_regression', LogisticRegression(
                    C=0.5, penalty='l1', solver='liblinear', random_state=42))
            )
        else:
            loans_model.steps.append(
                ('modelo_logistic_regression', LogisticRegression(
                    C=2.0, penalty='l2', solver='lbfgs', random_state=42))
            )

    elif mejormodelo == svm_acc:
        # Determinar la configuración específica de SVM
        if svm_acc == svm_acc_1:
            loans_model.steps.append(
                ('modelo_svm', SVC(C=1.0, kernel='linear', random_state=42))
            )
        elif svm_acc == svm_acc_2:
            loans_model.steps.append(
                ('modelo_svm', SVC(C=0.5, kernel='rbf', gamma=0.1, random_state=42))
            )
        else:
            loans_model.steps.append(
                ('modelo_svm', SVC(C=2.0, kernel='poly', degree=3, random_state=42))
            )

    elif mejormodelo == ada_acc:
        # Determinar la configuración específica de AdaBoost
        if ada_acc == adaboost_acc_1:
            loans_model.steps.append(
                ('modelo_ada_boost', AdaBoostClassifier(n_estimators=50, learning_rate=1.0,
                                                        random_state=42))
            )
        elif ada_acc == adaboost_acc_2:
            loans_model.steps.append(
                ('modelo_ada_boost', AdaBoostClassifier(n_estimators=100, learning_rate=0.5,
                                                        random_state=42))
            )
        else:
            loans_model.steps.append(
                ('modelo_ada_boost', AdaBoostClassifier(n_estimators=150, learning_rate=0.3,
                                                        random_state=42))
            )

    else:
        # Determinar la configuración específica de XGBoost
        if xgb_acc == xgb_acc_1:
            loans_model.steps.append(
                ('modelo_xgb', xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.3, random_state=42,
                    use_label_encoder=False))
            )
        elif xgb_acc == xgb_acc_2:
            loans_model.steps.append(
                ('modelo_xgb', xgb.XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                    colsample_bytree=0.8, random_state=42, use_label_encoder=False))
            )
        else:
            loans_model.steps.append(
                ('modelo_xgb', xgb.XGBClassifier(
                    n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.6,
                    colsample_bytree=0.6, gamma=1.0, random_state=42, use_label_encoder=False))
            )

    # Mostrar resultados
    print(f"El mejor modelo tiene una precisión de {mejormodelo:.4f}")
    print(f"Pipeline final: {loans_model}")

    with open('../artifacts/pipeline.pkl','wb') as f:
        pickle.dump(loans_model,f)
