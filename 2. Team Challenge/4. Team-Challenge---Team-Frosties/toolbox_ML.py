# toolbox_ML.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def describe_df(df):
    """
    Genera una descripción del DataFrame con información sobre las columnas.

    Argumentos:
    df (pd.DataFrame): DataFrame que se desea describir.

    Retorna:
    pd.DataFrame: DataFrame con información sobre cada columna:
        - Nombre de la columna
        - Tipo de dato
        - Porcentaje de valores nulos
        - Cantidad de valores únicos
        - Porcentaje de cardinalidad
    """
    description = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [df[col].dtype for col in df.columns],
        "Null Percentage (%)": [(df[col].isnull().sum() / len(df)) * 100 for col in df.columns],
        "Unique Values": [df[col].nunique() for col in df.columns],
        "Cardinality (%)": [(df[col].nunique() / len(df)) * 100 for col in df.columns],
    })
    return description

def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Clasifica las variables de un DataFrame en tipos sugeridos.

    Argumentos:
    df (pd.DataFrame): DataFrame cuyas variables se desean clasificar.
    umbral_categoria (int): Umbral para considerar una variable como categórica.
    umbral_continua (float): Umbral para considerar una variable como continua.

    Retorna:
    pd.DataFrame: DataFrame con columnas:
        - "nombre_variable": Nombre de las columnas originales.
        - "tipo_sugerido": Tipo sugerido para cada variable.
    """
    resultado = []

    for col in df.columns:
        cardinalidad = df[col].nunique()
        porcentaje_cardinalidad = cardinalidad / len(df)

        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        elif porcentaje_cardinalidad >= umbral_continua:
            tipo = "Numérica Continua"
        else:
            tipo = "Numérica Discreta"

        resultado.append({"nombre_variable": col, "tipo_sugerido": tipo})

    return pd.DataFrame(resultado)

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Obtiene columnas numéricas del DataFrame cuya correlación con una columna objetivo supera un umbral.

    Argumentos:
    df (pd.DataFrame): DataFrame con los datos.
    target_col (str): Nombre de la columna objetivo (debe ser numérica).
    umbral_corr (float): Umbral de correlación (valor absoluto) entre 0 y 1.
    pvalue (float, opcional): Nivel de significancia estadística para el test de correlación.

    Retorna:
    list: Lista de columnas que cumplen las condiciones de correlación y significancia.
    """
    if not (0 <= umbral_corr <= 1):
        print("Error: umbral_corr debe estar entre 0 y 1.")
        return None

    if pvalue is not None and not (0 <= pvalue <= 1):
        print("Error: pvalue debe estar entre 0 y 1 o ser None.")
        return None

    if target_col not in df.columns or not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: target_col no es una columna numérica válida del DataFrame.")
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = []

    for col in numeric_cols:
        if col != target_col:
            corr, p_val = pearsonr(df[target_col].dropna(), df[col].dropna())
            if abs(corr) > umbral_corr:
                if pvalue is None or p_val <= pvalue:
                    correlations.append(col)

    return correlations

def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Genera pairplots de las columnas numéricas seleccionadas en base a la correlación con una columna objetivo.

    Argumentos:
    df (pd.DataFrame): DataFrame con los datos.
    target_col (str): Nombre de la columna objetivo.
    columns (list of str): Lista de columnas a considerar.
    umbral_corr (float): Umbral de correlación (valor absoluto).
    pvalue (float, opcional): Nivel de significancia estadística para el test de correlación.

    Retorna:
    list: Lista de columnas que cumplen las condiciones de correlación y significancia.
    """
    if not isinstance(columns, list):
        print("Error: columns debe ser una lista.")
        return None

    if not (0 <= umbral_corr <= 1):
        print("Error: umbral_corr debe estar entre 0 y 1.")
        return None

    if pvalue is not None and not (0 <= pvalue <= 1):
        print("Error: pvalue debe estar entre 0 y 1 o ser None.")
        return None

    if target_col not in df.columns or not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: target_col no es una columna numérica válida del DataFrame.")
        return None

    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    valid_columns = []

    for col in columns:
        if col != target_col:
            corr, p_val = pearsonr(df[target_col].dropna(), df[col].dropna())
            if abs(corr) > umbral_corr:
                if pvalue is None or p_val <= pvalue:
                    valid_columns.append(col)

    if not valid_columns:
        print("No se encontraron columnas que cumplan las condiciones especificadas.")
        return []

    # Dividir las columnas en grupos de máximo 5 para los pairplots
    max_columns = 5
    valid_columns = [target_col] + valid_columns

    for i in range(1, len(valid_columns), max_columns - 1):
        subset = valid_columns[:1] + valid_columns[i:i + max_columns - 1]
        sns.pairplot(df[subset].dropna(), diag_kind="kde")
        plt.show()

    return valid_columns[1:]
