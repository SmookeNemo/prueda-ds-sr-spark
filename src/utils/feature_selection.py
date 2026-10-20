"""Selección de variables en PySpark: Information Value, correlación, VIF e importancia.

Todo se calcula SOLO con train.
- Set árboles (GBT, RF): IV + importancia. La multicolinealidad no afecta la predicción.
- Set logit: IV + filtro de correlación + VIF iterativo.
"""
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

from .eda import matriz_correlacion


# ----------------------------------------------------------------- Information Value
def _splits(sdf: DataFrame, col: str, bins: int, error_relativo: float = 0.001) -> list:
    """Cortes por cuantiles. Si hay muchos empates (variables infladas en ceros), agrega cortes
    calculados sobre los valores mayores al empate, para no juntar todo en un solo grupo."""
    probs = [i / bins for i in range(1, bins)]
    cortes = sorted(set(sdf.approxQuantile(col, probs, error_relativo)))
    if len(cortes) + 1 < max(2, bins // 2) and cortes:
        resto = sdf.filter(F.col(col) > max(cortes))
        if resto.limit(1).count():
            cortes = sorted(set(cortes) | set(resto.approxQuantile(col, probs, error_relativo)))
    return cortes


def _columna_grupo(sdf: DataFrame, col: str, bins: int, n_unicos: int):
    """Expresión con el grupo de cada fila: valor (discretas) o intervalo de cuantiles (continuas)."""
    c = F.col(col)
    if not isinstance(sdf.schema[col].dataType, NumericType) or n_unicos <= bins:
        return F.when(c.isNull(), F.lit("NULO")).otherwise(c.cast("string"))
    cortes = _splits(sdf, col, bins)
    expr = None
    limites = [-np.inf] + cortes + [np.inf]
    for i in range(len(limites) - 1):
        etiqueta = f"{i:02d}: [{limites[i]:g}, {limites[i + 1]:g})"
        cond = c < limites[i + 1]
        expr = F.when(cond, F.lit(etiqueta)) if expr is None else expr.when(cond, F.lit(etiqueta))
    return F.when(c.isNull(), F.lit("NULO")).otherwise(expr.otherwise(F.lit(f"{len(limites) - 2:02d}")))


def calcular_iv(sdf: DataFrame, feature: str, target: str, bins: int = 10, n_unicos: int = None):
    """IV de una variable. Devuelve (iv, tabla_woe)."""
    if n_unicos is None:
        n_unicos = sdf.select(F.countDistinct(feature)).first()[0]
    g = _columna_grupo(sdf, feature, bins, n_unicos)
    t = (sdf.groupBy(g.alias("grupo"))
            .agg(F.count("*").alias("n"), F.sum(target).alias("eventos"))
            .toPandas().set_index("grupo").sort_index())
    t["no_eventos"] = t["n"] - t["eventos"]
    # +0.5 evita divisiones por cero / log(0) en grupos sin eventos
    t["dist_eventos"] = (t["eventos"] + 0.5) / (t["eventos"].sum() + 0.5)
    t["dist_no_eventos"] = (t["no_eventos"] + 0.5) / (t["no_eventos"].sum() + 0.5)
    t["woe"] = np.log(t["dist_no_eventos"] / t["dist_eventos"])
    t["iv"] = (t["dist_no_eventos"] - t["dist_eventos"]) * t["woe"]
    t["tasa_evento"] = t["eventos"] / t["n"]
    return float(t["iv"].sum()), t


def _poder_iv(iv: float) -> str:
    if iv < 0.02:
        return "Inútil"
    if iv < 0.1:
        return "Débil"
    if iv < 0.3:
        return "Medio"
    if iv < 0.5:
        return "Fuerte"
    return "Sospechoso (revisar leakage)"


def tabla_iv(sdf: DataFrame, features: list, target: str, bins: int = 10) -> pd.DataFrame:
    """IV de todas las variables, ordenado de mayor a menor."""
    unicos = sdf.agg(*[F.countDistinct(f).alias(f) for f in features]).first().asDict()
    filas = [{"variable": f, "iv": calcular_iv(sdf, f, target, bins, unicos[f])[0]} for f in features]
    t = pd.DataFrame(filas).sort_values("iv", ascending=False).reset_index(drop=True)
    t["poder"] = t["iv"].apply(_poder_iv)
    return t


def filtrar_por_iv(t_iv: pd.DataFrame, min_iv: float = 0.02, max_iv: float = 0.5):
    """Devuelve (variables que pasan, variables sospechosas por IV alto)."""
    pasan = t_iv.loc[t_iv["iv"] >= min_iv, "variable"].tolist()
    sospechosas = t_iv.loc[t_iv["iv"] > max_iv, "variable"].tolist()
    print(f"Pasan IV >= {min_iv}: {len(pasan)} de {len(t_iv)}")
    if sospechosas:
        print(f"OJO, IV > {max_iv} (posible leakage): {sospechosas}")
    return pasan, sospechosas


# ----------------------------------------------------------------- Correlación (set logit)
def filtrar_correlacion(sdf: DataFrame, num_features: list, umbral: float = 0.8, prioridad: dict = None,
                        metodo: str = "spearman"):
    """Para cada par con |corr| > umbral elimina la de menor prioridad (por defecto el IV).
    Devuelve (variables que quedan, tabla de pares eliminados)."""
    if len(num_features) < 2:
        return list(num_features), pd.DataFrame()
    prioridad = prioridad or {}
    corr = matriz_correlacion(sdf, num_features, metodo).abs()
    pares = [(a, b, corr.loc[a, b]) for i, a in enumerate(num_features)
             for b in num_features[i + 1:] if corr.loc[a, b] > umbral]
    pares.sort(key=lambda p: -p[2])
    eliminadas, log = set(), []
    for a, b, r in pares:
        if a in eliminadas or b in eliminadas:
            continue
        quitar = a if prioridad.get(a, 0) < prioridad.get(b, 0) else b
        eliminadas.add(quitar)
        log.append({"var_1": a, "var_2": b, "corr": round(r, 3), "eliminada": quitar})
    quedan = [f for f in num_features if f not in eliminadas]
    print(f"Correlación > {umbral}: eliminadas {len(eliminadas)} -> {sorted(eliminadas)}")
    return quedan, pd.DataFrame(log)


# ----------------------------------------------------------------- VIF (set logit)
def _vif_desde_corr(corr: pd.DataFrame) -> pd.DataFrame:
    """VIF_i = elemento i de la diagonal de la inversa de la matriz de correlación (Pearson).
    Es equivalente a 1 / (1 - R²) de regresar cada variable contra las demás."""
    inv = np.linalg.pinv(corr.values)
    return (pd.DataFrame({"variable": corr.index, "vif": np.diag(inv)})
              .sort_values("vif", ascending=False).reset_index(drop=True))


def calcular_vif(sdf: DataFrame, num_features: list) -> pd.DataFrame:
    return _vif_desde_corr(matriz_correlacion(sdf, num_features, "pearson"))


def vif_iterativo(sdf: DataFrame, num_features: list, umbral: float = 5.0):
    """Quita una a una la variable de mayor VIF hasta que todas queden <= umbral.
    La matriz de correlación se calcula una sola vez en Spark; el resto es álgebra local.
    Devuelve (variables que quedan, tabla VIF final, variables eliminadas)."""
    if len(num_features) < 2:
        return list(num_features), pd.DataFrame(), []
    corr_total = matriz_correlacion(sdf, num_features, "pearson")
    feats, eliminadas = list(num_features), []
    while len(feats) > 1:
        t = _vif_desde_corr(corr_total.loc[feats, feats])
        if t["vif"].iloc[0] <= umbral:
            break
        peor = t["variable"].iloc[0]
        eliminadas.append((peor, round(float(t["vif"].iloc[0]), 2)))
        feats.remove(peor)
    print(f"VIF > {umbral}: eliminadas {eliminadas}")
    tabla = _vif_desde_corr(corr_total.loc[feats, feats]) if len(feats) > 1 else pd.DataFrame()
    return feats, tabla, eliminadas


# ----------------------------------------------------------------- Importancia (set árboles)
def importancia_arbol(sdf: DataFrame, num_features: list, cat_features: list, target: str,
                      weight_col: str = None, seed: int = 42) -> pd.DataFrame:
    """Importancia con un GBT rápido de Spark (equivalente al XGBoost del proyecto en pandas)."""
    imp_cols = [f"{c}__imp" for c in num_features]
    idx_cols = [f"{c}__idx" for c in cat_features]
    stages = []
    if num_features:
        stages.append(Imputer(strategy="median", inputCols=num_features, outputCols=imp_cols))
    if cat_features:
        stages.append(StringIndexer(inputCols=cat_features, outputCols=idx_cols, handleInvalid="keep"))
    stages.append(VectorAssembler(inputCols=imp_cols + idx_cols, outputCol="features", handleInvalid="keep"))
    gbt = GBTClassifier(labelCol=target, featuresCol="features", maxIter=50, maxDepth=4, seed=seed)
    if weight_col:
        gbt.setWeightCol(weight_col)
    modelo = Pipeline(stages=stages + [gbt]).fit(sdf)
    imp = modelo.stages[-1].featureImportances.toArray()
    t = pd.DataFrame({"variable": list(num_features) + list(cat_features), "importancia": imp})
    t["importancia"] = t["importancia"] / max(t["importancia"].sum(), 1e-12)
    t = t.sort_values("importancia", ascending=False).reset_index(drop=True)
    t["acumulado"] = t["importancia"].cumsum()
    return t


def seleccionar_por_importancia(imp: pd.DataFrame, acumulado: float = 0.95, min_importancia: float = 0.0):
    """Variables que explican el `acumulado` de la importancia total."""
    n = min(int((imp["acumulado"] <= acumulado).sum()) + 1, len(imp))
    sel = imp.head(n)
    sel = sel.loc[sel["importancia"] > min_importancia, "variable"].tolist()
    print(f"Importancia: {len(sel)} variables explican ~{acumulado:.0%}")
    return sel
