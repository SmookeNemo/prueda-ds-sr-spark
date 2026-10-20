"""Balanceo por pesos en PySpark. Se calcula UNA vez con train y se agrega como columna;
todos los modelos de Spark ML la usan con weightCol="peso".

Peso de la clase c:  w_c = n / (k * n_c)
Los pesos solo se usan para entrenar; las métricas se calculan sin ponderar.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

COL_PESO = "peso"


def pesos_por_clase(sdf: DataFrame, target: str) -> dict:
    conteos = {r[0]: r[1] for r in sdf.groupBy(target).count().collect()}
    n, k = sum(conteos.values()), len(conteos)
    return {c: round(n / (k * nc), 4) for c, nc in sorted(conteos.items())}


def agregar_pesos(sdf: DataFrame, target: str, col: str = COL_PESO):
    """Agrega la columna de pesos. Devuelve (sdf_con_pesos, pesos_por_clase)."""
    pesos = pesos_por_clase(sdf, target)
    expr = None
    for clase, w in pesos.items():
        expr = (F.when(F.col(target) == clase, F.lit(w)) if expr is None
                else expr.when(F.col(target) == clase, F.lit(w)))
    print(f"Pesos por clase: {pesos}")
    return sdf.withColumn(col, expr.otherwise(F.lit(1.0))), pesos
