"""Tratamiento de atípicos con IQR en PySpark.

- Los límites se calculan SOLO con train (approxQuantile).
- Capping: se materializa como un SQLTransformer (stage nativo de Spark ML) con los límites
  de train fijos. Va dentro del Pipeline, así que se guarda con el modelo en MLflow y se aplica
  igual en validation, test y producción. No requiere clases propias para serializar.
- Eliminar: quita filas de train (validation y test no se tocan).
"""
import pandas as pd
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

SUFIJO_CAP = "__cap"


def limites_iqr(sdf: DataFrame, cols: list, factor: float = 1.5, error_relativo: float = 0.001) -> dict:
    """{col: (lim_inf, lim_sup)} o {col: None} si IQR = 0 (no se trata: quedaría constante)."""
    if not cols:
        return {}
    qs = sdf.approxQuantile(cols, [0.25, 0.75], error_relativo)
    lims = {}
    for c, q in zip(cols, qs):
        if len(q) < 2:          # columna sin datos
            lims[c] = None
            continue
        iqr = q[1] - q[0]
        lims[c] = None if iqr == 0 else (q[0] - factor * iqr, q[1] + factor * iqr)
    return lims


def resumen_atipicos(sdf: DataFrame, num_cols: list, factor: float = 1.5) -> pd.DataFrame:
    """Límites IQR y cantidad / % de atípicos por variable (para el EDA)."""
    lims = limites_iqr(sdf, num_cols, factor)
    aggs = [F.count("*").alias("n")]
    for c, l in lims.items():
        if l is not None:
            aggs += [F.sum((F.col(c) < l[0]).cast("int")).alias(f"{c}__b"),
                     F.sum((F.col(c) > l[1]).cast("int")).alias(f"{c}__a")]
    r = sdf.agg(*aggs).first().asDict()
    filas = []
    for c, l in lims.items():
        if l is None:
            filas.append({"variable": c, "lim_inf": None, "lim_sup": None, "n_bajo": 0, "n_alto": 0,
                          "pct_atipicos": 0.0, "nota": "IQR=0: no se aplica tratamiento"})
        else:
            nb, na = r[f"{c}__b"], r[f"{c}__a"]
            filas.append({"variable": c, "lim_inf": l[0], "lim_sup": l[1], "n_bajo": nb, "n_alto": na,
                          "pct_atipicos": round(100 * (nb + na) / r["n"], 2), "nota": ""})
    return pd.DataFrame(filas).sort_values("pct_atipicos", ascending=False).reset_index(drop=True)


def eliminar_atipicos_iqr(sdf: DataFrame, cols: list, target: str, factor: float = 1.5):
    """Elimina las FILAS con algún valor fuera de los límites IQR. Usar SOLO sobre train.
    Columnas con IQR = 0 se omiten. Los nulos no cuentan como atípicos.
    Devuelve (sdf_filtrado, tabla_resumen)."""
    lims = limites_iqr(sdf, cols, factor)
    cond_ok = F.lit(True)
    aggs = [F.count("*").alias("n"), F.mean(target).alias("tasa")]
    for c, l in lims.items():
        if l is None:
            continue
        fuera = F.col(c).isNotNull() & ((F.col(c) < l[0]) | (F.col(c) > l[1]))
        aggs.append(F.sum(fuera.cast("int")).alias(c))
        cond_ok = cond_ok & ~fuera
    antes = sdf.agg(*aggs).first().asDict()
    filtrado = sdf.filter(cond_ok)
    despues = filtrado.agg(F.count("*").alias("n"), F.mean(target).alias("tasa")).first()
    resumen = pd.DataFrame([{"variable": c, "filas_atipicas": antes.get(c, 0),
                             "nota": "IQR=0: omitida" if l is None else ""} for c, l in lims.items()])
    resumen = resumen.sort_values("filas_atipicas", ascending=False).reset_index(drop=True)
    elim = antes["n"] - despues["n"]
    print(f"Filas eliminadas: {elim} de {antes['n']} ({100 * elim / antes['n']:.2f}%)")
    print(f"Tasa de evento: {antes['tasa']:.4f} -> {despues['tasa']:.4f}")
    return filtrado, resumen


def crear_capping(limites: dict, cols: list, sufijo_entrada: str = ""):
    """Construye el capping a partir de límites ya calculados con train (limites_iqr).
    Devuelve (SQLTransformer | None, {col: nombre_columna_salida}).
    Cada columna tratada genera '<col><sufijo_entrada>__cap' = LEAST(GREATEST(x, lim_inf), lim_sup).
    Columnas sin límites (no elegidas o con IQR = 0) salen sin cambios."""
    salida, exprs = {}, []
    for c in cols:
        entrada = f"{c}{sufijo_entrada}"
        l = limites.get(c)
        if l is None:
            salida[c] = entrada
            continue
        out = f"{entrada}{SUFIJO_CAP}"
        exprs.append(f"LEAST(GREATEST(`{entrada}`, {float(l[0])!r}), {float(l[1])!r}) AS `{out}`")
        salida[c] = out
    if not exprs:
        return None, salida
    return SQLTransformer(statement=f"SELECT *, {', '.join(exprs)} FROM __THIS__"), salida
