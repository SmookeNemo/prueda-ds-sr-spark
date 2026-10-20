"""Preparación específica del dataset (web_phishing). La usan train.py y predict.py,
así los datos nuevos reciben exactamente las mismas transformaciones."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

TARGET = "phishing"           # 1 = phishing, 0 = legítima
COLS_EXCLUIR = []             # IDs o variables con leakage (en este dataset no hay)


def leer_datos(spark, ruta: str) -> DataFrame:
    return spark.read.csv(ruta, header=True, inferSchema=True)


def preparar_features(sdf: DataFrame) -> DataFrame:
    """Excluye columnas y pasa los conteos a double (Imputer y VectorAssembler trabajan con double)."""
    sdf = sdf.drop(*[c for c in COLS_EXCLUIR if c in sdf.columns])
    return sdf.select(*[F.col(c).cast("double").alias(c) if c != TARGET else F.col(c) for c in sdf.columns])


def preparar_target(sdf: DataFrame) -> DataFrame:
    return sdf.withColumn(TARGET, F.col(TARGET).cast("int"))
