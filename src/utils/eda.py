"""EDA con PySpark. Los cálculos se hacen en Spark; solo los resultados agregados
(o una muestra, para los gráficos) se pasan a pandas."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType


def resumen_datos(sdf: DataFrame) -> pd.DataFrame:
    """Tipo, nulos, % nulos, valores únicos, % de ceros, mínimo, media y máximo por columna."""
    n = sdf.count()
    n_dup = n - sdf.dropDuplicates().count()
    print(f"Shape: ({n}, {len(sdf.columns)}) | Duplicados: {n_dup} ({100 * n_dup / max(n, 1):.2f}%)")
    tipos = dict(sdf.dtypes)
    aggs = []
    for c in sdf.columns:
        aggs += [F.sum(F.col(c).isNull().cast("int")).alias(f"{c}__nulos"),
                 F.countDistinct(c).alias(f"{c}__unicos")]
        if isinstance(sdf.schema[c].dataType, NumericType):
            aggs += [F.sum((F.col(c) == 0).cast("int")).alias(f"{c}__ceros"),
                     F.min(c).alias(f"{c}__min"), F.mean(c).alias(f"{c}__media"), F.max(c).alias(f"{c}__max")]
    r = sdf.agg(*aggs).first().asDict()
    filas = []
    for c in sdf.columns:
        filas.append({"variable": c, "tipo": tipos[c], "n_nulos": r[f"{c}__nulos"],
                      "pct_nulos": round(100 * r[f"{c}__nulos"] / n, 2), "n_unicos": r[f"{c}__unicos"],
                      "pct_ceros": round(100 * (r.get(f"{c}__ceros") or 0) / n, 2) if f"{c}__ceros" in r else None,
                      "min": r.get(f"{c}__min"), "media": r.get(f"{c}__media"), "max": r.get(f"{c}__max")})
    return pd.DataFrame(filas).set_index("variable")


def distribucion_target(sdf: DataFrame, target: str) -> pd.DataFrame:
    """Conteo y % por clase, más el ratio de desbalance."""
    t = sdf.groupBy(target).count().toPandas().set_index(target).sort_index()
    t["pct"] = (100 * t["count"] / t["count"].sum()).round(2)
    print(f"Ratio de desbalance (mayoritaria/minoritaria): {t['count'].max() / t['count'].min():.2f} : 1")
    return t.rename(columns={"count": "n"})


def tipos_variables(sdf: DataFrame, target: str, excluir=None):
    """Separa columnas numéricas y categóricas según el tipo de dato de Spark."""
    excluir = set(excluir or []) | {target}
    num_cols = [c for c in sdf.columns if c not in excluir and isinstance(sdf.schema[c].dataType, NumericType)]
    cat_cols = [c for c in sdf.columns if c not in excluir and c not in num_cols]
    print(f"Numéricas ({len(num_cols)}): {num_cols}")
    print(f"Categóricas ({len(cat_cols)}): {cat_cols}")
    return num_cols, cat_cols


def detectar_columnas_problema(sdf: DataFrame, umbral_nulos: float = 0.6) -> dict:
    """Columnas constantes, tipo ID (todo único) y con demasiados nulos."""
    n = sdf.count()
    r = sdf.agg(*[F.countDistinct(c).alias(f"{c}__u") for c in sdf.columns],
                *[F.sum(F.col(c).isNull().cast("int")).alias(f"{c}__n") for c in sdf.columns]).first().asDict()
    res = {
        "constantes": [c for c in sdf.columns if r[f"{c}__u"] <= 1],
        "tipo_id": [c for c in sdf.columns if r[f"{c}__u"] == n],
        "muchos_nulos": [c for c in sdf.columns if r[f"{c}__n"] / n > umbral_nulos],
    }
    for k, v in res.items():
        print(f"{k}: {v}")
    return res


def tasa_target_por_categoria(sdf: DataFrame, col: str, target: str) -> pd.DataFrame:
    """n por valor y tasa del evento (target=1). Sirve para categóricas y numéricas discretas."""
    return (sdf.groupBy(col).agg(F.count("*").alias("n"), F.mean(target).alias("tasa_evento"))
               .orderBy(F.desc("tasa_evento")).toPandas().set_index(col))


def tasa_target_por_presencia(sdf: DataFrame, num_cols: list, target: str) -> pd.DataFrame:
    """Para variables de conteo con muchos ceros: tasa del evento cuando la variable es 0 vs > 0."""
    aggs = []
    for c in num_cols:
        aggs += [F.sum((F.col(c) > 0).cast("int")).alias(f"{c}__n1"),
                 F.mean(F.when(F.col(c) > 0, F.col(target))).alias(f"{c}__t1"),
                 F.mean(F.when(F.col(c) == 0, F.col(target))).alias(f"{c}__t0")]
    r = sdf.agg(F.count("*").alias("n"), F.mean(target).alias("base"), *aggs).first().asDict()
    filas = [{"variable": c, "pct_mayor_0": round(100 * r[f"{c}__n1"] / r["n"], 2),
              "tasa_si_0": r[f"{c}__t0"], "tasa_si_mayor_0": r[f"{c}__t1"]} for c in num_cols]
    t = pd.DataFrame(filas).set_index("variable")
    t["lift_mayor_0"] = (t["tasa_si_mayor_0"] / r["base"]).round(2)
    print(f"Tasa base: {r['base']:.4f}")
    return t.sort_values("lift_mayor_0", ascending=False)


def graficar_numericas_vs_target(sdf: DataFrame, num_cols: list, target: str, ncols: int = 4,
                                 n_muestra: int = 20000, log: bool = True, seed: int = 42):
    """Boxplot por clase con una muestra (los gráficos se hacen en pandas).
    log=True usa escala log(1+x), útil con conteos de colas largas."""
    frac = min(1.0, n_muestra / max(sdf.count(), 1))
    pdf = sdf.select(*num_cols, target).sample(fraction=frac, seed=seed).toPandas()
    nrows = int(np.ceil(len(num_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, num_cols):
        datos = np.log1p(pdf[c].clip(lower=0)) if log else pdf[c]
        sns.boxplot(x=pdf[target], y=datos, ax=ax)
        ax.set_title(f"{c}{' (log1p)' if log else ''}")
        ax.set_ylabel("")
    for ax in axes[len(num_cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def matriz_correlacion(sdf: DataFrame, num_cols: list, metodo: str = "spearman") -> pd.DataFrame:
    """Matriz de correlación calculada en Spark (pearson o spearman)."""
    vec = VectorAssembler(inputCols=num_cols, outputCol="_v", handleInvalid="skip").transform(sdf).select("_v")
    m = Correlation.corr(vec, "_v", metodo).head()[0].toArray()
    return pd.DataFrame(m, index=num_cols, columns=num_cols)


def graficar_correlacion(sdf: DataFrame, num_cols: list, metodo: str = "spearman", figsize=(11, 9)):
    corr = matriz_correlacion(sdf, num_cols, metodo)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=len(num_cols) <= 20, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                ax=ax, annot_kws={"size": 7})
    ax.set_title(f"Correlación ({metodo})")
    fig.tight_layout()
    return fig


def pares_correlacionados(sdf: DataFrame, num_cols: list, umbral: float = 0.7, metodo: str = "spearman"):
    """Lista de pares con |correlación| > umbral."""
    corr = matriz_correlacion(sdf, num_cols, metodo)
    pares = [(a, b, corr.loc[a, b]) for i, a in enumerate(num_cols) for b in num_cols[i + 1:]
             if abs(corr.loc[a, b]) > umbral]
    return pd.DataFrame(pares, columns=["var_1", "var_2", "corr"]).sort_values("corr", key=abs, ascending=False)


def muestreo_estratificado(sdf: DataFrame, target: str, frac: float, seed: int = 42) -> DataFrame:
    """Muestra estratificada por el target (misma fracción por clase)."""
    clases = [r[0] for r in sdf.select(target).distinct().collect()]
    muestra = sdf.sampleBy(target, fractions={c: frac for c in clases}, seed=seed)
    print(f"Muestra: {muestra.count()} filas")
    return muestra
