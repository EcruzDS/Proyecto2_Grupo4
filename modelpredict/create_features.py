"""
    En este documento se realiza un proceso de ingenería de características
    que permiten realizar una limpieza de los datos y procesamiento para
    modelar datos.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_missing_data(dataset):
    """_summary_

    Args:
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    missing_columns = [
        (col, dataset[col].isnull().mean() * 100)
        for col in dataset.columns if dataset[col].isnull().any()
    ]
    if missing_columns:
        missing_df = pd.DataFrame(missing_columns, columns=['Columna', 'Porcentaje Faltante'])
        return missing_df
    return None


def plot_outliers_analysis(dataset, col):
    """_summary_

    Args:
        dataset (_type_): _description_
        col (_type_): _description_
    """
    plt.figure(figsize=(15, 6))
    print(f"Análisis de outliers para la variable: {col}")
    plt.subplot(131)
    sns.histplot(dataset[col], bins=30)
    plt.title("Densidad - Histograma")

    plt.subplot(132)
    stats.probplot(dataset[col], dist="norm", plot=plt)
    plt.title("QQ-Plot")

    plt.subplot(133)
    sns.boxplot(y=dataset[col])
    plt.title("Boxplot")
    plt.show()


def calculate_outlier_limits(dataset, col, multiplier=1.75):
    """_summary_

    Args:
        dataset (_type_): _description_
        col (_type_): _description_
        multiplier (float, optional): _description_. Defaults to 1.75.

    Returns:
        _type_: _description_
    """
    iqr = dataset[col].quantile(0.75) - dataset[col].quantile(0.25)
    lower_limit = dataset[col].quantile(0.25) - (multiplier * iqr)
    upper_limit = dataset[col].quantile(0.75) + (multiplier * iqr)
    return lower_limit, upper_limit


def handle_outliers_with_capping(dataset, col, multiplier=1.75):
    """_summary_

    Args:
        dataset (_type_): _description_
        col (_type_): _description_
        multiplier (float, optional): _description_. Defaults to 1.75.

    Returns:
        _type_: _description_
    """
    lower_limit, upper_limit = calculate_outlier_limits(dataset, col, multiplier)
    return np.where(dataset[col] > upper_limit, upper_limit,
                    np.where(dataset[col] < lower_limit, lower_limit, dataset[col]))


#data set utilizado
DATASET_PATH = '../data/raw/DataLoans_0424.csv'
db_loans = pd.read_csv(DATASET_PATH, delimiter=";")

replacements = {
    '1_BANCA INMOBILIARIA': '1_INMOBILIARIA',
    '2_BANCA PRIVADA': '2_NEGOCIO',
    '3_BANCA COMERCIAL': '3_COMERCIAL',
    '4_ASOCIACIONES Y OTROS': '4_RETAIL',
    '5_EMPRESAS CMI': '5_INTERCOMPANY'
}
db_loans['SegmentoComercial'] = db_loans['SegmentoComercial'].replace(replacements)

# datos sfaltantes
missing_data = analyze_missing_data(db_loans)
if missing_data is not None:
    print(missing_data)

# Tranamiento de Outliers
db_loans['Saldo$_capped'] = handle_outliers_with_capping(db_loans, "Saldo$")
