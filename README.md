# Evaluación técnica – Ciencia de Datos con PySpark

Clasificación de URLs de phishing (`web_phishing.csv`) con **PySpark + Spark ML + MLflow**,
siguiendo el mismo procedimiento del proyecto en pandas: EDA, split estratificado, atípicos por IQR,
pesos por clase, selección de variables (IV, correlación, VIF, importancia), challenge de modelos,
tuneo, umbral óptimo, registro del campeón y flujo de predicción.

## Requisitos
- **Python 3.10 – 3.13**
- **Java 17 o 21** (Spark 4 no funciona con Java 8 u 11). Verificar con `java -version`.
- **Windows:** además, `winutils.exe` y `hadoop.dll` (ver abajo).

## Instalación
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (PowerShell)
# source .venv/bin/activate         # Mac / Linux
pip install -r requirements.txt
python -m src.verificar_spark       # comprueba Java, Spark y el guardado de modelos
```

### Windows: Java y winutils (una sola vez)
1. Instalar Java 17 (por ejemplo Eclipse Temurin) y definir `JAVA_HOME` apuntando a la carpeta de instalación.
2. Descargar `winutils.exe` y `hadoop.dll` de la carpeta `hadoop-3.3.6/bin` del repositorio
   https://github.com/cdarlint/winutils y copiarlos a `C:\hadoop\bin`.
3. Variables de entorno: `HADOOP_HOME = C:\hadoop` y agregar `%HADOOP_HOME%\bin` al `PATH`.
4. Cerrar y volver a abrir VS Code, y correr `python -m src.verificar_spark`.

Sin winutils, Spark entrena pero falla al **guardar** modelos (y MLflow necesita guardarlos).

## Cómo ejecutar
```bash
python -m src.train                                   # challenge + tuneo + registro del campeón
python -m src.registrar                               # (opcional) listar runs
python -m src.registrar --run-id <id>                 # (opcional) registrar otro modelo como @champion
python -m src.predict data/nuevos.csv outputs/predicciones.csv
mlflow ui --backend-store-uri sqlite:///mlflow.db     # ver runs y modelo registrado
```

## Estructura
```
src/
├── features.py           # lectura y preparación del dataset (target, tipos)
├── train.py              # flujo supervisado completo (editar solo CONFIG)
├── predict.py            # flujo de ejecución con models:/<nombre>@champion
├── registrar.py          # registro manual de un run como campeón
├── verificar_spark.py    # chequeo del entorno
└── utils/
    ├── spark_session.py  # SparkSession local
    ├── eda.py            # resumen, target, presencia (0 vs >0), correlación, gráficos con muestra
    ├── outliers.py       # límites IQR, eliminar filas, capping como SQLTransformer
    ├── feature_selection.py  # IV, correlación, VIF (inversa de la matriz de correlación), importancia GBT
    ├── weights.py        # columna de pesos por clase (weightCol)
    ├── threshold.py      # umbral óptimo por iteración (una pasada con bins de probabilidad)
    ├── metrics.py        # matriz de confusión completa, AUC, Gini, KS, PR-AUC, gaps, gráficos
    ├── modeling.py       # split estratificado, Pipeline de Spark ML, challenge, CrossValidator
    └── mlflow_utils.py   # log de runs, registro del campeón, carga y predicción
```

## Equivalencias con el proyecto en pandas
| pandas / sklearn | PySpark / Spark ML |
|---|---|
| `train_test_split(stratify=y)` | `randomSplit` por clase y `unionByName` |
| `SimpleImputer` | `Imputer` |
| `IQRCapper` (clase propia) | `SQLTransformer` con `LEAST/GREATEST` y límites de train |
| `OneHotEncoder` | `StringIndexer` + `OneHotEncoder` (solo logit; los árboles usan el índice) |
| `StandardScaler` | `VectorAssembler` + `StandardScaler` |
| `sample_weight` | columna `peso` + `weightCol` |
| XGBoost | `GBTClassifier` |
| `RandomizedSearchCV` | `CrossValidator` + `ParamGridBuilder` con folds estratificados (`foldCol`) |
| `mlflow.sklearn` | `mlflow.spark` |

## Decisiones metodológicas
- Split estratificado 70/15/15 **antes** de cualquier transformación (sin leakage).
- Duplicados: se mantienen (son URLs distintas con los mismos conteos); ver EDA.
- Atípicos: configurables por variable (eliminar filas de train, capping o sin tratamiento).
- Selección: IV en train (IV < 0.02 fuera, IV > 0.5 revisar leakage).
  - Árboles: IV + importancia acumulada 95 %.
  - Logit: IV + correlación > 0.8 + VIF > 5.
- Balanceo: columna de pesos por clase, la misma para todos los modelos.
- Umbral: búsqueda por iteración en validation (maximiza F1); Youden como referencia.
- Métricas en train/val/test con umbral 0.5 y óptimo; gaps de AUC y F1 para sobreajuste.
- Campeón: mejor AUC en validation entre modelos con gap AUC train-val ≤ 0.05; en empate, el más simple.
- Solo el campeón se registra en MLflow (alias `champion`, umbral como tag).

## Tiempos
En una máquina de 2 núcleos el entrenamiento completo tarda ~25 min (la mayor parte es el tuneo).
Para acelerar: `FRACCION_TUNEO` más bajo, `TUNEAR_TOP = 1` o `GUARDAR_MODELOS_CHALLENGE = False`.

## Resultados (corrida de referencia)
| Modelo | AUC val | Gini val | KS val | F1 val (umbral óptimo) | Gap AUC train-val |
|---|---|---|---|---|---|
| random_forest_tuned | 0.956 | 0.911 | 0.772 | 0.845 | 0.004 |
| gbt | 0.953 | 0.906 | 0.770 | 0.841 | 0.002 |
| gbt_tuned | 0.953 | 0.906 | 0.770 | 0.841 | 0.002 |
| random_forest | 0.950 | 0.899 | 0.765 | 0.840 | 0.002 |
| logit | 0.934 | 0.868 | 0.720 | 0.809 | 0.000 |

**Campeón:** `random_forest_tuned`, umbral óptimo 0.65.
**Test:** AUC 0.955 | Gini 0.909 | KS 0.767 | Precision 82.8 % | Recall 86.1 % | F1 0.844
