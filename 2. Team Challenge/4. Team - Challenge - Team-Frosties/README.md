# TEAM CHALLENGE: TOOLBOX (I) 🧰🛠️

Este repositorio contiene la primera parte del *Team Challenge* dedicado a construir un módulo de herramientas básicas para la preparación de modelos de *Machine Learning* de manera más sencilla.

## **Índice**   
1. [Objetivo](#Objetivo)
2. [Contenido del repositorio](#Contenido-del-repositorio)
3. [Instalacion](#Instalacion)
4. [Instrucciones de uso](#Instrucciones-de-uso)
5. [Requisitos](#Requisitos)
6. [Funciones a implementar](#Funciones-a-implementar)
7. [Contribuciones](#Contribuciones)
8. [Autores](#Autores)

## Objetivo

El objetivo principal de este proyecto es crear un conjunto de funciones esenciales para el preprocesamiento y análisis de datos, las cuales serán utilizadas en la segunda parte del reto para resolver un problema de *Machine Learning*.

## Contenido del repositorio

Este repositorio incluye lo siguiente:

1. **Script `toolbox_ML.py`:**
    - Contiene el conjunto de funciones necesarias especificadas en la sección correspondiente del notebook.
    - El código está comentado adecuadamente para facilitar su comprensión y uso.

2. **Ejemplo de uso:**
    - Un notebook o script demostrativo donde se muestra la aplicación de las funciones desarrolladas.

## Instalacion 

- clonar el repositorio o descargar el archivo [aqui](https://github.com/rafaneda7/Team-Challenge---Team-Frosties)
- ejecutar el programa:
python toolbox_ML.py

## Instrucciones de uso

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/rafaneda7/Team-Challenge---Team-Frosties.git
   ```
2. Importar el script en tu proyecto:
   ```python
   from toolbox_ML import funcion_nombre
   ```
3. Ejecutar el notebook de ejemplo para ver las funciones en acción.

## Requisitos

- Python 3.x
- Librerías necesarias (especificadas en el script y notebook de ejemplo):
  - NumPy
  - Pandas
  - Scikit-learn

## Funciones a implementar

A continuación, se detallan las funciones que se deben crear y su propósito:

1. **`describe_df(df: pd.DataFrame) -> pd.DataFrame`**
   - Genera un resumen del DataFrame que incluye el tipo de dato de cada columna, el porcentaje de valores nulos, los valores únicos y el porcentaje de cardinalidad.

2. **`tipifica_variables(df: pd.DataFrame, umbral_categoria: int, umbral_continua: float) -> pd.DataFrame`**
   - Clasifica las variables en "Binaria", "Categórica", "Numérica Continua" o "Numérica Discreta" según su cardinalidad y los umbrales proporcionados.

3. **`get_features_num_regression(df: pd.DataFrame, target_col: str, umbral_corr: float, pvalue: float = None) -> list`**
   - Devuelve una lista de columnas numéricas cuya correlación con la variable objetivo supere un valor de umbral y, si se especifica, pase una prueba de significancia estadística.

4. **`plot_features_num_regression(df: pd.DataFrame, target_col: str, columns: list = [], umbral_corr: float = 0, pvalue: float = None) -> None`**
   - Genera gráficos de pares entre la variable objetivo y las variables numéricas seleccionadas, agrupando de cinco en cinco si son muchas.

5. **`get_features_cat_regression(df: pd.DataFrame, target_col: str, pvalue: float = 0.05) -> list`**
   - Devuelve una lista de columnas categóricas cuya relación con la variable objetivo sea significativa según una prueba estadística apropiada.

6. **`plot_features_cat_regression(df: pd.DataFrame, target_col: str, columns: list = [], pvalue: float = 0.05, with_individual_plot: bool = False) -> None`**
   - Genera histogramas de las variables categóricas seleccionadas en relación con la variable objetivo, con opción de gráficos individuales.

## Contribuciones

Las contribuciones al proyecto están abiertas. Por favor, sigue los siguientes pasos para colaborar:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`feature/nueva-funcion`).
3. Realiza los cambios y haz un commit.
4. Envía un pull request para revisión.

---
## Autores

- [MiguelAngel120](https://github.com/MiguelAngel120)
- [joaquinvillarmaldonado](https://github.com/joaquinvillarmaldonado)
- [MarcoFuchs98](https://github.com/MarcoFuchs98)
- [Johannkarl](https://github.com/Johannkarl)
- [rafaneda7](https://github.com/rafaneda7)

*Equipo de trabajo: The Bridge Data Science Bootcamp*