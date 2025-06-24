# 📊 Lab 14 - Minería de Datos

# Predicción de Productividad de Empleados de Confección

## 📋 Descripción

Este proyecto implementa un análisis completo de machine learning para predecir la productividad de empleados en el sector de confección textil. Utiliza el dataset "Productivity Prediction of Garment Employees" del repositorio UCI Machine Learning para desarrollar y comparar diferentes modelos de ensamblaje.

## 🎯 Objetivos

- Realizar preprocesamiento completo de datos incluyendo tratamiento de valores faltantes y outliers
- Implementar técnicas de ensamblaje: Voting, Bagging, Boosting y Stacking
- Comparar modelos base: k-NN, SVM, Regresión Lineal, Árboles de Decisión y Random Forest
- Evaluar el rendimiento usando múltiples métricas de regresión

## 🔧 Instalación

### Requisitos

```bash
pip install pandas numpy polars ucimlrepo scikit-learn pytest pyarrow
```

### Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/productivity-prediction-garment-employees.git
cd productivity-prediction-garment-employees
```

## 📊 Dataset

**Fuente**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees)

**Características**:
- **Variable objetivo**: `actual_productivity`
- **Variables eliminadas**: `date` (como se especifica en los requisitos)
- **Tipos de variables**: Numéricas y categóricas
- **Tamaño**: 1,197 observaciones

## 🏗️ Estructura del Proyecto

```
├── README.md
├── utils.py                 # Funciones de preprocesamiento y modelado
├── test_utils.py            # Tests unitarios
├── main.ipynb              # Notebook principal con análisis completo
├── requirements.txt        # Dependencias del proyecto
└── results/               # Resultados y visualizaciones
    ├── model_comparison.csv
    └── plots/
```

## 🔄 Metodología

### 1. Preprocesamiento de Datos

- **Valores faltantes**: Imputación con mediana
- **Outliers**: Tratamiento univariado con método IQR (capping)
- **Variables categóricas**: Conversión a variables dummy
- **Escalado**: StandardScaler para variables numéricas
- **División**: 80% entrenamiento, 20% prueba

### 2. Modelos Base

| Modelo | Hiperparámetros |
|--------|----------------|
| **Regresión Lineal** | Configuración por defecto |
| **Random Forest** | n_estimators=100, max_depth=10 |
| **k-NN** | n_neighbors=5 |
| **SVM** | kernel='rbf', C=10.0, gamma='scale' |
| **Árbol de Decisión** | max_depth=8 |

### 3. Técnicas de Ensamblaje

- **Voting Regressor**: Combina Regresión Lineal, Random Forest y k-NN
- **Bagging**: Múltiples árboles de decisión
- **Boosting**: Gradient Boosting con 200 estimadores
- **Stacking**: Random Forest, k-NN y SVM con Regresión Lineal como meta-modelo

### 4. Métricas de Evaluación

- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score** (Coeficiente de Determinación)

## 🚀 Uso

### Ejecución completa

```python
# Importar utilidades
import utils

# Cargar y preprocesar datos
df = utils.load_and_prepare_data()
df = utils.handle_missing_values(df)

# Identificar tipos de columnas
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
df = utils.convert_to_dummies(df, cat_cols)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols.remove('actual_productivity')
df = utils.treat_outliers_iqr(df, num_cols)

# División y escalado
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop(columns='actual_productivity')
y = df['actual_productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Entrenamiento y evaluación
results = utils.train_and_evaluate_models(X_train, y_train, X_test, y_test, 42)
print(results.sort_values(by="R2 Score", ascending=False))
```

### Ejecutar tests

```bash
pytest test_utils.py -v
```

## 📈 Resultados Esperados

Los modelos se evalúan y ordenan por R² Score (coeficiente de determinación). Típicamente se espera:

1. **Stacking**: Mejor rendimiento general
2. **Boosting**: Segundo mejor rendimiento
3. **Random Forest/Bagging**: Rendimiento intermedio
4. **Voting**: Rendimiento variable según modelos base

## 🧪 Testing

El proyecto incluye tests unitarios completos que cubren:

- Manejo de valores faltantes
- Tratamiento de outliers
- Conversión a variables dummy
- Validación de tipos de datos
- Casos edge (listas vacías, sin valores faltantes)

```bash
# Ejecutar todos los tests
pytest test_utils.py

# Ejecutar con coverage
pytest test_utils.py --cov=utils
```

## 📝 Notas Técnicas

### Decisiones de Diseño

- **Polars para manipulación**: Eliminación eficiente de la columna 'date'
- **Logging integrado**: Seguimiento del proceso de entrenamiento
- **Manejo de errores**: Tratamiento robusto de excepciones durante entrenamiento
- **Reproducibilidad**: Semilla fija (RANDOM_STATE=42) para resultados consistentes

### Mejoras Implementadas

- Solución del FutureWarning de pandas en imputación
- Hiperparámetros optimizados para mejor rendimiento
- Tests más robustos con múltiples casos de prueba
- Logging detallado para debugging

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE.md](LICENSE.md) para detalles.

## 👥 Autor

- GitHub: https://github.com/OlorteguiKevin
- Email: kevinolorteguip503@gmail.com

## 🙏 Agradecimientos

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) por proporcionar el dataset
- Scikit-learn por las herramientas de machine learning
- Comunidad de Python por las librerías utilizadas

## 📚 Referencias

- Imran, A.A., Amin, M.N., Rifat, M.R.I. (2019). Deep Neural Network Approach for Predicting the Productivity of Garment Employees. IEEE Region 10 Symposium (TENSYMP).
- UCI Machine Learning Repository: Productivity Prediction of Garment Employees Data Set

