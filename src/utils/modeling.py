"""Split, pipelines de Spark ML, challenge de modelos y tuneo de hiperparámetros."""
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .metrics import calcular_gaps, diagnostico_sobreajuste, evaluar_modelo
from .outliers import crear_capping
from .threshold import COL_PROB, agregar_prob, buscar_umbral, umbral_youden
from .weights import COL_PESO


# ----------------------------------------------------------------- Split
def split_train_val_test(sdf: DataFrame, target: str, val_size: float = 0.15, test_size: float = 0.15,
                         seed: int = 42):
    """Split estratificado (70/15/15 por defecto): se divide cada clase por separado y se une.
    Hacerlo ANTES de cualquier transformación. Devuelve (train, val, test) cacheados."""
    sdf = sdf.cache()
    pesos = [1 - val_size - test_size, val_size, test_size]
    partes = {"train": None, "val": None, "test": None}
    for clase in sorted(r[0] for r in sdf.select(target).distinct().collect()):
        trozos = sdf.filter(F.col(target) == clase).randomSplit(pesos, seed=seed)
        for nombre, trozo in zip(partes, trozos):
            partes[nombre] = trozo if partes[nombre] is None else partes[nombre].unionByName(trozo)
    for nombre in partes:
        partes[nombre] = partes[nombre].cache()
        r = partes[nombre].agg(F.count("*").alias("n"), F.mean(target).alias("t")).first()
        print(f"{nombre:5s}: {r['n']:>8} filas | tasa evento {r['t']:.4f}")
    return partes["train"], partes["val"], partes["test"]


# ----------------------------------------------------------------- Pipeline
def construir_pipeline(modelo, num_cols: list, cat_cols: list, escalar: bool = True,
                       limites_capping: dict = None, one_hot: bool = None) -> Pipeline:
    """Numéricas: Imputer (mediana) -> capping IQR (SQLTransformer, opcional) -> [escalado].
    Categóricas: StringIndexer -> [One-Hot]. Luego VectorAssembler -> modelo.
    limites_capping: {col: (lim_inf, lim_sup)} calculados con train; las columnas que no estén
    (o con None) no se recortan. one_hot por defecto = escalar (logit sí, árboles no)."""
    one_hot = escalar if one_hot is None else one_hot
    stages, entradas = [], []
    if num_cols:
        stages.append(Imputer(strategy="median", inputCols=list(num_cols),
                              outputCols=[f"{c}__imp" for c in num_cols]))
        sql, salida = crear_capping(limites_capping or {}, list(num_cols), sufijo_entrada="__imp")
        if sql is not None:
            stages.append(sql)
        entradas += [salida[c] for c in num_cols]
    if cat_cols:
        idx = [f"{c}__idx" for c in cat_cols]
        stages.append(StringIndexer(inputCols=list(cat_cols), outputCols=idx, handleInvalid="keep"))
        if one_hot:
            ohe = [f"{c}__ohe" for c in cat_cols]
            stages.append(OneHotEncoder(inputCols=idx, outputCols=ohe, handleInvalid="keep"))
            entradas += ohe
        else:
            entradas += idx
    col_vec = "features_raw" if escalar else "features"
    stages.append(VectorAssembler(inputCols=entradas, outputCol=col_vec, handleInvalid="keep"))
    if escalar:
        stages.append(StandardScaler(inputCol=col_vec, outputCol="features", withMean=True, withStd=True))
    modelo.setFeaturesCol("features")
    return Pipeline(stages=stages + [modelo])


def modelos_base(target: str, seed: int = 42) -> dict:
    """Modelos candidatos de Spark ML, todos con weightCol (pesos por clase).
    GBT es el equivalente nativo de Spark al XGBoost del proyecto en pandas."""
    comunes = dict(labelCol=target, weightCol=COL_PESO)
    return {
        "logit": LogisticRegression(maxIter=100, **comunes),
        "random_forest": RandomForestClassifier(numTrees=200, maxDepth=8, minInstancesPerNode=20,
                                                seed=seed, **comunes),
        "gbt": GBTClassifier(maxIter=100, maxDepth=4, stepSize=0.1, subsamplingRate=0.8, seed=seed, **comunes),
    }


# ----------------------------------------------------------------- Challenge
def entrenar_y_evaluar(nombre: str, pipeline, data: dict, label: str, metrica_umbral: str = "f1") -> dict:
    """Entrena (si recibe un Pipeline) o usa el PipelineModel ya entrenado, busca el umbral en
    validation y evalúa en train/val/test. data = {'train': sdf_con_pesos, 'val': sdf, 'test': sdf}."""
    modelo = pipeline if isinstance(pipeline, PipelineModel) else pipeline.fit(data["train"])
    p_val = agregar_prob(modelo.transform(data["val"])).select(label, COL_PROB).cache()
    umbral_opt, _, _, _ = buscar_umbral(p_val, label, metrica=metrica_umbral)
    youden = umbral_youden(p_val, label)
    p_val.unpersist()
    df_eval = evaluar_modelo(modelo, data, umbral_opt, label, nombre)
    gaps = calcular_gaps(df_eval)
    return {"nombre": nombre, "pipeline": modelo, "umbral": umbral_opt, "umbral_youden": youden,
            "metricas": df_eval, "gaps": gaps, "diagnostico": diagnostico_sobreajuste(gaps)}


def tabla_comparativa(resultados: dict) -> pd.DataFrame:
    """Una fila por modelo: métricas de validation con umbral óptimo + gaps + diagnóstico."""
    filas = []
    for nombre, r in resultados.items():
        m = r["metricas"]
        val_opt = m[(m["conjunto"] == "val") & (m["tipo_umbral"] == "optimo")].iloc[0]
        val_05 = m[(m["conjunto"] == "val") & (m["tipo_umbral"] == "0.5")].iloc[0]
        filas.append({
            "modelo": nombre, "umbral": r["umbral"],
            "auc_val": val_opt["auc"], "gini_val": val_opt["gini"], "ks_val": val_opt["ks"],
            "pr_auc_val": val_opt["pr_auc"],
            "f1_val_0.5": val_05["f1"], "f1_val_opt": val_opt["f1"],
            "recall_val_opt": val_opt["recall"], "precision_val_opt": val_opt["precision"],
            **{k: v for k, v in r["gaps"].items() if k.startswith(("gap_auc", "gap_f1"))},
            "diagnostico": r["diagnostico"],
        })
    return pd.DataFrame(filas).sort_values("auc_val", ascending=False).reset_index(drop=True)


def elegir_campeon(tabla: pd.DataFrame, metrica: str = "auc_val", max_gap: float = 0.05,
                   empate: float = 0.005, preferir_simple: str = "logit") -> str:
    """1) descarta modelos con gap AUC train-val > max_gap; 2) mejor `metrica` en validation;
    3) si el modelo simple está a menos de `empate` del mejor, se prefiere el simple."""
    validos = tabla[tabla["gap_auc_train_val"] <= max_gap]
    descartados = tabla.loc[tabla["gap_auc_train_val"] > max_gap, "modelo"].tolist()
    if descartados:
        print(f"Descartados por sobreajuste (gap AUC > {max_gap}): {descartados}")
    if validos.empty:
        print("Todos superan el gap máximo; se elige el de menor gap.")
        return tabla.sort_values("gap_auc_train_val").iloc[0]["modelo"]
    validos = validos.sort_values(metrica, ascending=False)
    mejor = validos.iloc[0]
    simple = validos[validos["modelo"] == preferir_simple]
    if mejor["modelo"] != preferir_simple and not simple.empty and mejor[metrica] - simple.iloc[0][metrica] <= empate:
        print(f"Empate técnico: se prefiere {preferir_simple} por simplicidad.")
        return preferir_simple
    return mejor["modelo"]


# ----------------------------------------------------------------- Tuneo
def espacios_busqueda(nombre: str, modelo) -> list:
    """Grilla de hiperparámetros (Spark no tiene búsqueda aleatoria: se usa una grilla chica).
    Debe construirse con la MISMA instancia de modelo que está dentro del Pipeline."""
    g = ParamGridBuilder()
    if nombre == "logit":
        g = g.addGrid(modelo.regParam, [0.0, 0.01, 0.1]).addGrid(modelo.elasticNetParam, [0.0])
    elif nombre == "random_forest":
        g = g.addGrid(modelo.maxDepth, [8, 12]).addGrid(modelo.minInstancesPerNode, [5, 20])
    elif nombre == "gbt":
        g = g.addGrid(modelo.maxDepth, [4, 6]).addGrid(modelo.stepSize, [0.1, 0.2])
    return g.build()


def agregar_folds(sdf: DataFrame, target: str, k: int = 3, seed: int = 42, col: str = "fold") -> DataFrame:
    """Folds estratificados: dentro de cada clase se asigna fold = posición aleatoria mod k."""
    w = Window.partitionBy(target).orderBy(F.rand(seed))
    return sdf.withColumn(col, (F.row_number().over(w) % k).cast("int"))


def tunear_modelo(pipeline: Pipeline, grid: list, train: DataFrame, target: str, folds: int = 3,
                  seed: int = 42, paralelismo: int = 2, fraccion: float = 1.0):
    """CrossValidator con folds estratificados (foldCol) y AUC como métrica.
    Los pesos se usan en el entrenamiento de cada fold (weightCol).
    fraccion < 1: la búsqueda se hace sobre una muestra estratificada de train (más rápido) y el
    mejor modelo se reentrena con TODO train. Devuelve
    (mejor PipelineModel entrenado con todo train, mejores parámetros, mejor AUC CV)."""
    datos = train
    if fraccion < 1:
        clases = [r[0] for r in train.select(target).distinct().collect()]
        datos = train.sampleBy(target, {c: fraccion for c in clases}, seed=seed)
    train_f = agregar_folds(datos, target, folds, seed).cache()
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid,
                        evaluator=BinaryClassificationEvaluator(labelCol=target, metricName="areaUnderROC"),
                        foldCol="fold", parallelism=paralelismo, seed=seed)
    cv_modelo = cv.fit(train_f)
    i = int(np.argmax(cv_modelo.avgMetrics))
    mejores = {p.name: v for p, v in grid[i].items()}
    print(f"Mejor AUC CV: {cv_modelo.avgMetrics[i]:.4f} ({len(grid)} combinaciones x {folds} folds"
          f"{f', muestra {fraccion:.0%} de train' if fraccion < 1 else ''})")
    print(f"Mejores parámetros: {mejores}")
    train_f.unpersist()
    mejor = cv_modelo.bestModel if fraccion >= 1 else pipeline.fit(train, params=grid[i])
    return mejor, mejores, float(cv_modelo.avgMetrics[i])
