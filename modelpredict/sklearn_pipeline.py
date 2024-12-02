"""Este modulo contiene las funciones del pipeline para transformar el dataset
"""
import pickle
import configparser
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import CountFrequencyEncoder
from feature_engine.encoding import OrdinalEncoder
from feature_engine.outliers import Winsorizer
def pipeline_model():
    """Esta función crea la ingeniería de características del modelo
    """
    dataset = pd.read_csv('../data/raw/DataLoans_0424.csv', delimiter = ";")
    dataset.head()
    label_encoder = LabelEncoder()
    dataset['SegmentoComercial'] = label_encoder.fit_transform(dataset['SegmentoComercial'])
    dataset.head()
    config = configparser.ConfigParser()
    config.read('../pipeline.cfg')
    drop_vars = list(config.get('GENERAL','VARS_TO_DROP').split(','))
    drop_vars.append(config.get('GENERAL','TARGET'))
    x_features = dataset.drop(labels = drop_vars, axis = 1)
    y_target =dataset[config.get('GENERAL','TARGET')]
    x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size = 0.3,
                                        shuffle = True, random_state = 2025)
    # Pipeline
    loans_model = Pipeline([
        # Frequency Encoding
        ('freq_var_encoding', CountFrequencyEncoder(encoding_method='count',
        variables=config.get('ENCODING','VARS_TO_FREQ_ENCODE').split(','))),
        # Label Encoding
        ('lab_var_encoding', OrdinalEncoder(encoding_method='ordered',
            variables=config.get('ENCODING', 'VARS_TO_LABEL_ENCODE').split(',')
        )),
        # Método Capping
        ('capping_var', Winsorizer(capping_method='iqr',tail='both',fold=1.5,
            variables=config.get('CAPPING', 'VARS_TO_CAPP').split(',')
        )),
        # Feature Scaling
        ('feature_scaling', StandardScaler())
    ])
    loans_model.fit(x_train,y_train)
    x_features_processed = loans_model.transform(x_train)
    df_features_processed = pd.DataFrame(x_features_processed, columns = x_train.columns)
    df_features_processed['SegmentoComercial'] = y_train.reset_index()['SegmentoComercial']
    df_features_processed.to_csv('../data/processed/feature_for_models.csv',index=False)
    x_features_processed_test = loans_model.transform(x_test)
    df_features_processed_test = pd.DataFrame(x_features_processed_test, columns = x_test.columns)
    df_features_processed_test['SegmentoComercial'] = y_test.reset_index()['SegmentoComercial']
    df_features_processed_test.to_csv('../data/processed/test_dataset.csv',index=False)

    with open('../artifacts/pipeline.pkl','wb') as f:
        pickle.dump(loans_model,f)
