"""Métricas completas en PySpark, evaluación train/val/test con umbral 0.5 y óptimo,
gaps de sobreajuste y gráficos (ROC, matriz de confusión)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from sklearn.metrics import ConfusionMatrixDisplay

from .threshold import COL_PROB, agregar_prob, confusion_por_umbral, tabla_bins


def _div(a, b):
    return float(a / b) if b else 0.0


def curva_roc(pred: DataFrame, label: str, n_bins: int = 1000):
    """(fpr, tpr) a partir de bins de probabilidad (una pasada en Spark)."""
    cm = confusion_por_umbral(tabla_bins(pred, label, n_bins))
    P, N = cm["TP"].iloc[0] + cm["FN"].iloc[0], cm["FP"].iloc[0] + cm["TN"].iloc[0]
    tpr = np.r_[cm["TP"].to_numpy() / P, 0.0]
    fpr = np.r_[cm["FP"].to_numpy() / N, 0.0]
    return fpr, tpr


def metricas_curva(pred: DataFrame, label: str) -> dict:
    """Métricas que NO dependen del umbral: AUC, Gini, KS y PR-AUC."""
    ev = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol=COL_PROB)
    auc = ev.setMetricName("areaUnderROC").evaluate(pred)
    pr_auc = ev.setMetricName("areaUnderPR").evaluate(pred)
    fpr, tpr = curva_roc(pred, label)
    return {"auc": auc, "gini": 2 * auc - 1, "ks": float(np.max(tpr - fpr)), "pr_auc": pr_auc}


def metricas_confusion(pred: DataFrame, label: str, umbral: float = 0.5) -> dict:
    """Todas las métricas de la matriz de confusión (exactas, una pasada en Spark)."""
    y, p = F.col(label) == 1, F.col(COL_PROB) >= umbral
    r = pred.agg(F.sum((y & p).cast("int")).alias("TP"), F.sum((~y & p).cast("int")).alias("FP"),
                 F.sum((~y & ~p).cast("int")).alias("TN"), F.sum((y & ~p).cast("int")).alias("FN")).first()
    tp, fp, tn, fn = (int(r[k] or 0) for k in ("TP", "FP", "TN", "FN"))
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    return {"umbral": umbral, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy": _div(tp + tn, tp + tn + fp + fn), "precision": precision, "npv": _div(tn, tn + fn),
            "recall": recall, "especificidad": _div(tn, tn + fp), "fpr": _div(fp, fp + tn),
            "fnr": _div(fn, fn + tp), "f1": _div(2 * precision * recall, precision + recall)}


def metricas_completas(pred: DataFrame, label: str, umbral: float = 0.5, curva: dict = None) -> dict:
    """Matriz de confusión + métricas de curva. `pred` debe tener la columna p1 (agregar_prob)."""
    return {**metricas_confusion(pred, label, umbral), **(curva or metricas_curva(pred, label))}


def evaluar_modelo(modelo, conjuntos: dict, umbral_opt: float, label: str, nombre: str = "") -> pd.DataFrame:
    """Evalúa un PipelineModel en cada conjunto ({'train': sdf, 'val': sdf, 'test': sdf})
    con umbral 0.5 y con el óptimo. Devuelve una fila por (conjunto, umbral)."""
    filas = []
    for set_name, sdf in conjuntos.items():
        pred = agregar_prob(modelo.transform(sdf)).select(label, COL_PROB).cache()
        curva = metricas_curva(pred, label)
        for tipo, u in [("0.5", 0.5), ("optimo", umbral_opt)]:
            filas.append({"modelo": nombre, "conjunto": set_name, "tipo_umbral": tipo,
                          **metricas_completas(pred, label, u, curva)})
        pred.unpersist()
    return pd.DataFrame(filas)


def calcular_gaps(df_eval: pd.DataFrame) -> dict:
    """Brechas de sobreajuste (umbral óptimo): train-val y val-test."""
    opt = df_eval[df_eval["tipo_umbral"] == "optimo"].set_index("conjunto")
    gaps = {}
    for met in ["auc", "f1", "ks"]:
        if {"train", "val"} <= set(opt.index):
            gaps[f"gap_{met}_train_val"] = opt.loc["train", met] - opt.loc["val", met]
        if {"val", "test"} <= set(opt.index):
            gaps[f"gap_{met}_val_test"] = opt.loc["val", met] - opt.loc["test", met]
    return gaps


def diagnostico_sobreajuste(gaps: dict, tolerancia: float = 0.05) -> str:
    g = gaps.get("gap_auc_train_val", 0)
    if g > 2 * tolerancia:
        return "Sobreajuste grave"
    if g > tolerancia:
        return "Sobreajuste moderado"
    return "OK"


# ----------------------------------------------------------------- Gráficos
def graficar_roc(curvas: dict, label: str, titulo: str = "Curva ROC"):
    """curvas: {'nombre': pred_sdf_con_p1, ...}. Varias curvas en el mismo gráfico."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for nombre, pred in curvas.items():
        fpr, tpr = curva_roc(pred, label)
        auc = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol=COL_PROB).evaluate(pred)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="gray")
    ax.set_xlabel("FPR (1 - especificidad)")
    ax.set_ylabel("TPR (recall)")
    ax.set_title(titulo)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def graficar_matriz_confusion(pred: DataFrame, label: str, umbral: float, titulo: str = ""):
    m = metricas_confusion(pred, label, umbral)
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(titulo or f"Matriz de confusión (umbral={umbral:.2f})")
    fig.tight_layout()
    return fig
