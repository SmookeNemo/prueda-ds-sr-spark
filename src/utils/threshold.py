"""Umbral óptimo de corte en PySpark. SIEMPRE se elige en validation, nunca en test.

Para no recorrer los datos una vez por umbral, se agrupan las probabilidades en bins
(una sola pasada en Spark) y las matrices de confusión de todos los umbrales se obtienen
con sumas acumuladas en pandas. Es exacto para umbrales múltiplos del paso (0.01, 0.02, ...).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

COL_PROB = "p1"


def agregar_prob(pred: DataFrame, col_probabilidad: str = "probability") -> DataFrame:
    """Extrae P(clase 1) del vector de probabilidades de Spark ML."""
    return pred.withColumn(COL_PROB, vector_to_array(F.col(col_probabilidad))[1])


def tabla_bins(pred: DataFrame, label: str, n_bins: int = 100) -> pd.DataFrame:
    """Positivos y negativos por bin de probabilidad (bin k = [k/n, (k+1)/n))."""
    b = F.least(F.floor(F.col(COL_PROB) * n_bins), F.lit(n_bins - 1)).cast("int")
    t = (pred.groupBy(b.alias("bin"))
             .agg(F.sum(F.col(label)).alias("pos"), F.count("*").alias("n"))
             .toPandas().set_index("bin").reindex(range(n_bins), fill_value=0))
    t["neg"] = t["n"] - t["pos"]
    return t[["pos", "neg"]].astype(float)


def confusion_por_umbral(bins: pd.DataFrame) -> pd.DataFrame:
    """TP, FP, TN, FN para cada umbral k/n (predicción positiva si p >= umbral)."""
    n_bins = len(bins)
    pos_ge = bins["pos"][::-1].cumsum()[::-1].to_numpy()   # positivos con bin >= k
    neg_ge = bins["neg"][::-1].cumsum()[::-1].to_numpy()
    P, N = bins["pos"].sum(), bins["neg"].sum()
    return pd.DataFrame({"umbral": np.arange(n_bins) / n_bins, "TP": pos_ge, "FP": neg_ge,
                         "FN": P - pos_ge, "TN": N - neg_ge})


def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def _metrica(cm: pd.DataFrame, metrica: str):
    prec, rec = _div(cm.TP, cm.TP + cm.FP), _div(cm.TP, cm.TP + cm.FN)
    if metrica == "f1":
        return _div(2 * prec * rec, prec + rec)
    if metrica == "f2":
        return _div(5 * prec * rec, 4 * prec + rec)
    if metrica == "precision":
        return prec
    if metrica == "recall":
        return rec
    if metrica == "especificidad":
        return _div(cm.TN, cm.TN + cm.FP)
    if metrica == "accuracy":
        return _div(cm.TP + cm.TN, cm.TP + cm.TN + cm.FP + cm.FN)
    if metrica == "youden":
        return rec - _div(cm.FP, cm.FP + cm.TN)
    raise ValueError(f"Métrica no soportada: {metrica}")


def buscar_umbral(pred: DataFrame, label: str, metrica: str = "f1", paso: float = 0.01):
    """Itera umbrales de `paso` a 1-paso y devuelve el que maximiza `metrica`
    ('f1', 'f2', 'precision', 'recall', 'accuracy', 'youden').
    Devuelve (umbral_opt, score_opt, umbrales, scores)."""
    n_bins = int(round(1 / paso))
    cm = confusion_por_umbral(tabla_bins(pred, label, n_bins)).iloc[1:]   # excluye umbral 0
    scores = _metrica(cm, metrica)
    i = int(np.argmax(scores))
    return float(round(cm["umbral"].iloc[i], 4)), float(scores[i]), cm["umbral"].to_numpy(), scores


def umbral_youden(pred: DataFrame, label: str, paso: float = 0.01) -> float:
    """Umbral que maximiza TPR - FPR (Youden J). Se reporta como referencia."""
    return buscar_umbral(pred, label, "youden", paso)[0]


def umbral_con_restriccion(pred: DataFrame, label: str, objetivo: str = "precision",
                           restriccion: str = "recall", minimo: float = 0.8, paso: float = 0.01):
    """Maximiza `objetivo` sujeto a `restriccion` >= minimo."""
    cm = confusion_por_umbral(tabla_bins(pred, label, int(round(1 / paso)))).iloc[1:]
    ok = _metrica(cm, restriccion) >= minimo
    if not ok.any():
        print("Ningún umbral cumple la restricción")
        return None, None
    obj = np.where(ok, _metrica(cm, objetivo), -np.inf)
    i = int(np.argmax(obj))
    return float(round(cm["umbral"].iloc[i], 4)), float(obj[i])


def graficar_umbral(pred: DataFrame, label: str, umbral_opt: float = None, paso: float = 0.01, titulo: str = ""):
    """Precision, recall y F1 en función del umbral, marcando el óptimo y el 0.5."""
    cm = confusion_por_umbral(tabla_bins(pred, label, int(round(1 / paso)))).iloc[1:]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cm["umbral"], _metrica(cm, "precision"), label="Precision")
    ax.plot(cm["umbral"], _metrica(cm, "recall"), label="Recall")
    ax.plot(cm["umbral"], _metrica(cm, "f1"), label="F1", lw=2.5)
    ax.axvline(0.5, ls=":", c="gray", label="Umbral 0.5")
    if umbral_opt is not None:
        ax.axvline(umbral_opt, ls="--", c="red", label=f"Óptimo = {umbral_opt:.2f}")
    ax.set_xlabel("Umbral")
    ax.set_ylabel("Métrica (validation)")
    ax.set_title(titulo or "Métricas vs umbral")
    ax.legend()
    fig.tight_layout()
    return fig
