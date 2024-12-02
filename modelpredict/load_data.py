"""
Este módulo carga datos de un archivo CSV para su análisis. Incluye validación de la ruta
y manejo de errores al cargar el archivo.
"""

# ##### **Universidad Galileo**
# ##### **Postgrado en Análisis y Predicción de Datos** - *Cuarto Trimestre 2024*
# ##### **Product Development**
# ##### **Alumno:** Elder Cruz. **Carnet:** 23004456
# ##### **Alumno:** Walter Reyes. **Carnet:** 23004450
# ##### **Alumno:** Dayana Gamboa. **Carnet:** 23001913
# ##### **Alumno:** Rodrigo Cano. **Carnet:** 23001916
# ##### **Proyecto final:** Carga de datos

# # 1. Importación de librerías

import pandas as pd

# # 2. Carga de dataset

DATASET_PATH  ='../data/raw/DataLoans_0424.csv'

DB_loans = pd.read_csv(DATASET_PATH, delimiter = ";")
DB_loans.head()
