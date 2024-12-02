"""Este modulo genera el pipeline de entrenamiento del modelo
"""
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb


def train_pipeline_model():
    """Este modulo genera las predicciones y elige el mejor modelo en el pipeline para clasificar
    """
    train_data = pd.read_csv('../data/processed/feature_for_models.csv')
    test_data = pd.read_csv('../data/processed/test_dataset.csv')
    with open('../artifacts/pipeline.pkl','rb') as f:
        loans_model = pickle.load(f)
    # Agregar lectura de target con el archivo de configuración
    x_features = train_data.drop(labels=['SegmentoComercial'], axis = 1)
    y_target = train_data['SegmentoComercial']

    x_features_test = test_data.drop(labels=['SegmentoComercial'], axis = 1)
    y_target_test = test_data['SegmentoComercial']

    rf_model = RandomForestClassifier().fit(x_features,y_target)
    y_preds = rf_model.predict(x_features_test)

    rf_acc = accuracy_score(y_preds, y_target_test)
    lr_model = LogisticRegression().fit(x_features,y_target)
    y_preds = lr_model.predict(x_features_test)

    lr_acc = accuracy_score(y_preds, y_target_test)
    svm_model = SVC().fit(x_features,y_target)
    y_preds = svm_model.predict(x_features_test)
    svm_acc = accuracy_score(y_preds, y_target_test)
    ada_model = AdaBoostClassifier().fit(x_features,y_target)
    y_preds = ada_model.predict(x_features_test)
    ada_acc = accuracy_score(y_preds, y_target_test)
    xgb_model = xgb.XGBClassifier().fit(x_features,y_target)
    y_preds = xgb_model.predict(x_features_test)
    xgb_acc = accuracy_score(y_preds, y_target_test)
    modelos = [rf_acc,lr_acc,svm_acc,ada_acc,xgb_acc]
    mejormodelo = max(modelos)

    if mejormodelo == rf_acc:
        loans_model.steps.append(
            ('modelo_random_forest',RandomForestClassifier()))

    elif mejormodelo == lr_acc:
        loans_model.steps.append(
            ('modelo_logistic_regression',LogisticRegression()))

    elif mejormodelo == svm_acc:
        loans_model.steps.append(
            ('modelo_svm',SVC()))

    elif mejormodelo == ada_acc:
        loans_model.steps.append(
            ('modelo_ada_boost',AdaBoostClassifier()))

    else:
        loans_model.steps.append(
            ('modelo_xgb',xgb.XGBClassifier()))
