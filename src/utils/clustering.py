"""Segmentación (no supervisado) con PySpark + Spark ML + MLflow.

Flujo: Imputer -> [capping IQR] -> log1p (colas largas) -> StandardScaler -> [PCA] -> algoritmo.
- El preprocesamiento se ajusta UNA vez con train y se reutiliza para todos los k y algoritmos.
- k se elige con silhouette (en validation) + método del codo + tamaño mínimo de cluster.
- Challenge: KMeans vs BisectingKMeans vs GaussianMixture con el mismo k.
- El campeón se guarda como un único PipelineModel (preprocesamiento + modelo).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import BisectingKMeans, GaussianMixture, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import PCA, Imputer, SQLTransformer, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .outliers import crear_capping

COL_CLUSTER = "cluster"
COL_VEC = "features"


# ----------------------------------------------------------------- Preprocesamiento
def construir_preprocesamiento(features: list, cols_log: list = None, limites_capping: dict = None,
                               n_pca: int = None) -> Pipeline:
    """Imputer (mediana) -> capping IQR (opcional) -> log1p -> StandardScaler -> PCA (opcional).
    La salida final es la columna 'features'."""
    cols_log = [c for c in (cols_log or []) if c in features]
    stages = [Imputer(strategy="median", inputCols=list(features), outputCols=[f"{c}__imp" for c in features])]
    sql_cap, salida = crear_capping(limites_capping or {}, list(features), sufijo_entrada="__imp")
    if sql_cap is not None:
        stages.append(sql_cap)
    if cols_log:
        exprs = [f"LN(1 + GREATEST(`{salida[c]}`, 0)) AS `{c}__log`" for c in cols_log]
        stages.append(SQLTransformer(statement=f"SELECT *, {', '.join(exprs)} FROM __THIS__"))
        salida.update({c: f"{c}__log" for c in cols_log})
    stages.append(VectorAssembler(inputCols=[salida[c] for c in features], outputCol="features_raw"))
    col_std = "features_std" if n_pca else COL_VEC
    stages.append(StandardScaler(inputCol="features_raw", outputCol=col_std, withMean=True, withStd=True))
    if n_pca:
        stages.append(PCA(k=n_pca, inputCol=col_std, outputCol=COL_VEC))
    return Pipeline(stages=stages)


def elegir_n_pca(sdf_train: DataFrame, features: list, cols_log: list = None, limites_capping: dict = None,
                 varianza: float = 0.85):
    """Número de componentes que explican al menos `varianza`. Devuelve (n, varianza_explicada, figura)."""
    prep = construir_preprocesamiento(features, cols_log, limites_capping, n_pca=len(features)).fit(sdf_train)
    var = prep.stages[-1].explainedVariance.toArray()
    acum = np.cumsum(var)
    n = int(np.searchsorted(acum, varianza) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(var) + 1), var, label="Varianza por componente")
    ax.plot(range(1, len(var) + 1), acum, marker="o", c="C1", label="Acumulada")
    ax.axhline(varianza, ls="--", c="gray")
    ax.axvline(n, ls="--", c="red", label=f"n = {n} ({acum[n - 1]:.0%})")
    ax.set_xlabel("Componente")
    ax.set_ylabel("Varianza explicada")
    ax.set_title("PCA: varianza explicada")
    ax.legend()
    fig.tight_layout()
    print(f"PCA: {n} componentes explican {acum[n - 1]:.1%} de la varianza")
    return n, var, fig


# ----------------------------------------------------------------- Modelos y métricas
def crear_modelo(algoritmo: str, k: int, seed: int = 42):
    comunes = dict(k=k, seed=seed, featuresCol=COL_VEC, predictionCol=COL_CLUSTER)
    if algoritmo == "kmeans":
        return KMeans(maxIter=50, initMode="k-means||", **comunes)
    if algoritmo == "bisecting_kmeans":
        return BisectingKMeans(maxIter=50, **comunes)
    if algoritmo == "gmm":
        return GaussianMixture(maxIter=100, probabilityCol="prob_cluster", **comunes)
    raise ValueError(f"Algoritmo no soportado: {algoritmo}")


def silhouette(pred: DataFrame) -> float:
    """Silhouette con distancia euclidiana al cuadrado (de -1 a 1; más alto = clusters más separados)."""
    return ClusteringEvaluator(featuresCol=COL_VEC, predictionCol=COL_CLUSTER, metricName="silhouette",
                               distanceMeasure="squaredEuclidean").evaluate(pred)


def tamanos(pred: DataFrame) -> pd.Series:
    t = pred.groupBy(COL_CLUSTER).count().toPandas().set_index(COL_CLUSTER)["count"].sort_index()
    return t / t.sum()


def _costo(modelo):
    """WSSSE (suma de distancias al cuadrado al centroide). GMM no lo tiene: devuelve NaN."""
    resumen = getattr(modelo, "summary", None)
    return float(getattr(resumen, "trainingCost", np.nan)) if resumen is not None else np.nan


def evaluar_k(train_t: DataFrame, val_t: DataFrame, k_rango=range(2, 11), algoritmo: str = "kmeans",
              seed: int = 42) -> pd.DataFrame:
    """Para cada k: silhouette en train y validation, WSSSE (codo) y tamaño del cluster más chico.
    train_t / val_t ya vienen preprocesados (columna 'features')."""
    filas = []
    for k in k_rango:
        m = crear_modelo(algoritmo, k, seed).fit(train_t)
        p_tr, p_val = m.transform(train_t), m.transform(val_t)
        tam = tamanos(p_tr)
        filas.append({"k": k, "silhouette_train": silhouette(p_tr), "silhouette_val": silhouette(p_val),
                      "wssse": _costo(m), "pct_cluster_min": float(tam.min()), "pct_cluster_max": float(tam.max())})
        print(f"k={k:2d} | silhouette val={filas[-1]['silhouette_val']:.4f} | "
              f"cluster más chico={filas[-1]['pct_cluster_min']:.1%}")
    return pd.DataFrame(filas)


def elegir_k(tabla: pd.DataFrame, k_min: int = 3, min_pct: float = 0.03) -> int:
    """Mejor silhouette en validation entre los k >= k_min cuyo cluster más chico tenga al menos min_pct.
    k = 2 suele dar la silhouette más alta pero es demasiado grueso para segmentar clientes."""
    cand = tabla[(tabla["k"] >= k_min) & (tabla["pct_cluster_min"] >= min_pct)]
    if cand.empty:
        print("Ningún k cumple las condiciones; se usa el de mayor silhouette.")
        cand = tabla
    k = int(cand.sort_values("silhouette_val", ascending=False).iloc[0]["k"])
    print(f"k elegido: {k}")
    return k


def challenge_algoritmos(train_t: DataFrame, val_t: DataFrame, k: int,
                         algoritmos=("kmeans", "bisecting_kmeans", "gmm"), seed: int = 42):
    """Compara algoritmos con el mismo k. Devuelve (tabla, {algoritmo: modelo})."""
    filas, modelos = [], {}
    for alg in algoritmos:
        m = crear_modelo(alg, k, seed).fit(train_t)
        p_tr, p_val = m.transform(train_t), m.transform(val_t)
        tam = tamanos(p_tr)
        s_tr, s_val = silhouette(p_tr), silhouette(p_val)
        filas.append({"algoritmo": alg, "k": k, "silhouette_train": s_tr, "silhouette_val": s_val,
                      "gap_silhouette": s_tr - s_val, "wssse": _costo(m),
                      "pct_cluster_min": float(tam.min()), "pct_cluster_max": float(tam.max()),
                      "n_clusters_reales": int((tam > 0).sum())})
        modelos[alg] = m
    tabla = pd.DataFrame(filas).sort_values("silhouette_val", ascending=False).reset_index(drop=True)
    return tabla, modelos


def elegir_campeon_cluster(tabla: pd.DataFrame, min_pct: float = 0.03, empate: float = 0.01,
                           preferir: str = "kmeans") -> str:
    """Mejor silhouette en validation entre algoritmos sin clusters diminutos;
    si K-Means está a menos de `empate` del mejor, se prefiere por simple e interpretable."""
    cand = tabla[tabla["pct_cluster_min"] >= min_pct]
    if cand.empty:
        cand = tabla
    mejor = cand.iloc[0]
    simple = cand[cand["algoritmo"] == preferir]
    if mejor["algoritmo"] != preferir and not simple.empty and \
            mejor["silhouette_val"] - simple.iloc[0]["silhouette_val"] <= empate:
        print(f"Empate técnico: se prefiere {preferir} por simplicidad e interpretabilidad.")
        return preferir
    return mejor["algoritmo"]


def estabilidad_semillas(train_t: DataFrame, algoritmo: str, k: int, semillas=(1, 2, 3, 4, 5)) -> pd.DataFrame:
    """Entrena con varias semillas y compara las asignaciones con el Adjusted Rand Index (ARI):
    1 = misma segmentación, ~0 = asignación al azar. ARI alto => segmentos estables."""
    from sklearn.metrics import adjusted_rand_score
    base = None
    filas = []
    datos = train_t.select(COL_VEC).cache()
    for s in semillas:
        etiquetas = crear_modelo(algoritmo, k, s).fit(datos).transform(datos).select(COL_CLUSTER).toPandas()[COL_CLUSTER]
        if base is None:
            base = etiquetas
            continue
        filas.append({"semilla_base": semillas[0], "semilla": s, "ari": adjusted_rand_score(base, etiquetas)})
    datos.unpersist()
    t = pd.DataFrame(filas)
    print(f"Estabilidad ({algoritmo}, k={k}): ARI medio = {t['ari'].mean():.3f} (mín {t['ari'].min():.3f})")
    return t


def unir_pipeline(prep_model: PipelineModel, modelo) -> PipelineModel:
    """Preprocesamiento ajustado + modelo de clustering = un solo PipelineModel para MLflow."""
    return PipelineModel(stages=list(prep_model.stages) + [modelo])


# ----------------------------------------------------------------- Perfilado
def perfilar_clusters(pred: DataFrame, variables: list) -> pd.DataFrame:
    """Media de cada variable ORIGINAL por cluster, más tamaño y % de clientes."""
    aggs = [F.count("*").alias("n_clientes")] + [F.mean(c).alias(c) for c in variables]
    perfil = pred.groupBy(COL_CLUSTER).agg(*aggs).toPandas().set_index(COL_CLUSTER).sort_index()
    perfil.insert(1, "pct_clientes", (100 * perfil["n_clientes"] / perfil["n_clientes"].sum()).round(1))
    return perfil


def perfil_relativo(pred: DataFrame, variables: list) -> pd.DataFrame:
    """Índice = media del cluster / media global (1 = promedio, 2 = el doble). Facilita nombrar segmentos."""
    perfil = perfilar_clusters(pred, variables)[variables]
    globales = pred.agg(*[F.mean(c).alias(c) for c in variables]).toPandas().iloc[0]
    return (perfil / globales.replace(0, np.nan)).round(2)


def rasgos_principales(rel: pd.DataFrame, top: int = 3, umbral: float = 0.25) -> pd.DataFrame:
    """Las variables más por encima y más por debajo del promedio en cada cluster.
    Solo se listan las que se alejan al menos `umbral` del promedio (índice > 1.25 o < 0.75)."""
    filas = []
    for cl, fila in rel.iterrows():
        orden = fila.dropna().sort_values()
        arriba = orden[orden > 1 + umbral].index[::-1][:top]
        abajo = orden[orden < 1 - umbral].index[:top]
        filas.append({COL_CLUSTER: cl,
                      "muy_por_encima": ", ".join(f"{v} (x{orden[v]:.1f})" for v in arriba) or "-",
                      "muy_por_debajo": ", ".join(f"{v} (x{orden[v]:.1f})" for v in abajo) or "-"})
    return pd.DataFrame(filas).set_index(COL_CLUSTER)


# ----------------------------------------------------------------- Gráficos
def graficar_codo_silhouette(tabla: pd.DataFrame, k_elegido: int = None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(tabla["k"], tabla["wssse"], marker="o")
    a1.set_title("Método del codo (WSSSE)")
    a1.set_xlabel("k")
    a2.plot(tabla["k"], tabla["silhouette_train"], marker="o", label="train")
    a2.plot(tabla["k"], tabla["silhouette_val"], marker="o", label="validation")
    a2.set_title("Silhouette")
    a2.set_xlabel("k")
    a2.legend()
    if k_elegido:
        for a in (a1, a2):
            a.axvline(k_elegido, ls="--", c="red")
    fig.tight_layout()
    return fig


def graficar_clusters_2d(pred: DataFrame, n_muestra: int = 5000, seed: int = 42):
    """Dispersión en los 2 primeros componentes del vector 'features' (PC1 y PC2 si se usó PCA)."""
    frac = min(1.0, n_muestra / max(pred.count(), 1))
    pdf = (pred.sample(fraction=frac, seed=seed)
               .select(vector_to_array(COL_VEC).alias("v"), COL_CLUSTER).toPandas())
    xy = np.vstack(pdf["v"].to_numpy())[:, :2]
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=pdf[COL_CLUSTER], cmap="tab10", s=8, alpha=0.6)
    ax.legend(*sc.legend_elements(), title="Cluster", loc="best")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.set_title("Clusters (primeros 2 componentes)")
    fig.tight_layout()
    return fig


def graficar_perfil(rel: pd.DataFrame):
    """Heatmap del índice vs promedio global (escala log2: 0 = promedio, +1 = el doble, -1 = la mitad)."""
    datos = np.log2(rel.clip(lower=0.05))
    fig, ax = plt.subplots(figsize=(max(10, 0.6 * rel.shape[1]), 0.8 * rel.shape[0] + 2))
    sns.heatmap(datos, annot=rel, fmt=".1f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "log2(índice)"})
    ax.set_title("Perfil de clusters (índice vs promedio global)")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    return fig
