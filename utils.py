import logging
from typing import List, Tuple
import pandas as pd
import numpy as np
import polars as pl
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, BaggingRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_prepare_data() -> pd.DataFrame:
    """
    Carga y prepara el dataset eliminando la columna 'date'.
    Returns:
        DataFrame listo para preprocesamiento.
    """
    data = fetch_ucirepo(id=597)
    df = pd.concat([data.data.features, data.data.targets], axis=1)
    df = pl.from_pandas(df).drop("date").to_pandas()
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores nulos usando la mediana.
    Versión mejorada que evita el FutureWarning de pandas.
    """
    df_copy = df.copy()  # Trabajamos con una copia explícita

    for col in df_copy.columns:
        if df_copy[col].isnull().sum() > 0:
            # Método recomendado que evita el warning
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())

    logging.info(f"Valores faltantes imputados en {df_copy.columns[df.isnull().any()].tolist()}")
    return df_copy

def treat_outliers_iqr(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """
    Aplica capping para outliers usando IQR.
    Versión mejorada con logging.
    """
    df_copy = df.copy()
    outliers_info = {}

    for col in num_cols:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Contar outliers antes del tratamiento
        outliers_count = ((df_copy[col] < lower) | (df_copy[col] > upper)).sum()
        outliers_info[col] = outliers_count

        df_copy[col] = np.clip(df_copy[col], lower, upper)

    logging.info(f"Outliers tratados: {outliers_info}")
    return df_copy

def convert_to_dummies(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """
    Convierte variables categóricas a dummies.
    Versión mejorada con validación.
    """
    if not cat_cols:
        logging.info("No hay variables categóricas para convertir")
        return df

    df_dummies = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)
    logging.info(f"Variables categóricas convertidas: {cat_cols}")
    logging.info(f"Nuevas columnas dummy: {[col for col in df_dummies.columns if col not in df.columns]}")

    return df_dummies

def train_and_evaluate_models(X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series,
                               random_state: int) -> pd.DataFrame:
    """
    Entrena y evalúa modelos ensamblados.
    Versión mejorada con mejor manejo de errores y logging.
    """
    # Modelos base con hiperparámetros mejorados
    r1 = LinearRegression()
    r2 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state)
    r3 = KNeighborsRegressor(n_neighbors=5)
    r4 = SVR(kernel='rbf', C=10.0, gamma='scale')
    r5 = DecisionTreeRegressor(max_depth=8, random_state=random_state)

    models = {
        'Voting': VotingRegressor([('lr', r1), ('rf', r2), ('knn', r3)]),
        'Bagging': BaggingRegressor(r5, n_estimators=100, random_state=random_state),
        'Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=random_state),
        'Stacking': StackingRegressor([('rf', r2), ('knn', r3), ('svr', r4)], final_estimator=r1)
    }

    results = []
    for name, model in models.items():
        try:
            logging.info(f"Entrenando modelo: {name}")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)

            results.append({
                "Modelo": name,
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "MAE": mean_absolute_error(y_test, pred),
                "R2 Score": r2_score(y_test, pred)
            })
            logging.info(f"Modelo {name} entrenado exitosamente")

        except Exception as e:
            logging.error(f"Error entrenando modelo {name}: {str(e)}")
            continue

    return pd.DataFrame(results)

def analyze_data_quality(df: pd.DataFrame) -> dict:
    """
    Función adicional para análisis de calidad de datos.
    """
    quality_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.to_dict()
    }

    # Estadísticas para variables numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        quality_report['numeric_stats'] = df[numeric_cols].describe().to_dict()

    return quality_report
