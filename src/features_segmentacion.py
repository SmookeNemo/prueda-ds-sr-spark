"""Preparación del dataset de segmentación (bank_customer_segmentation). La usan clustering.py
y segmentar.py, así los clientes nuevos reciben exactamente las mismas transformaciones."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

ID = "CUST_ID"

# Variables originales de comportamiento (TENURE se deja solo para el perfilado: el 85 % vale 12)
VARIABLES = ["BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
             "CASH_ADVANCE", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY",
             "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX",
             "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"]

# Variables derivadas (ratios de negocio, más fáciles de interpretar en el perfilado)
DERIVADAS = ["UTILIZACION_LIMITE", "RATIO_PAGO_MINIMO", "COMPRA_PROMEDIO_TRX"]

# Montos y conteos con cola larga: se transforman con log1p antes de escalar
COLS_LOG = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE",
            "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
            "RATIO_PAGO_MINIMO", "COMPRA_PROMEDIO_TRX"]


def leer_datos(spark, ruta: str) -> DataFrame:
    return spark.read.csv(ruta, header=True, inferSchema=True)


def preparar_features(sdf: DataFrame) -> DataFrame:
    """Limpieza basada en el EDA + variables derivadas. Los nulos restantes los imputa el Pipeline."""
    sdf = sdf.select(F.col(ID), *[F.col(c).cast("double").alias(c) for c in VARIABLES])
    sdf = (sdf
           # Frecuencia > 1 es un error de datos (máximo lógico = 1)
           .withColumn("CASH_ADVANCE_FREQUENCY", F.least(F.col("CASH_ADVANCE_FREQUENCY"), F.lit(1.0)))
           # Sin pagos registrados, el pago mínimo nulo significa 0 (no un dato faltante al azar)
           .withColumn("MINIMUM_PAYMENTS", F.when(F.col("MINIMUM_PAYMENTS").isNull() & (F.col("PAYMENTS") == 0), 0.0)
                       .otherwise(F.col("MINIMUM_PAYMENTS"))))
    return (sdf
            .withColumn("UTILIZACION_LIMITE", F.when(F.col("CREDIT_LIMIT") > 0,
                                                     F.col("BALANCE") / F.col("CREDIT_LIMIT")))
            .withColumn("RATIO_PAGO_MINIMO", F.when(F.col("MINIMUM_PAYMENTS") > 0,
                                                    F.col("PAYMENTS") / F.col("MINIMUM_PAYMENTS")))
            .withColumn("COMPRA_PROMEDIO_TRX", F.when(F.col("PURCHASES_TRX") > 0,
                                                      F.col("PURCHASES") / F.col("PURCHASES_TRX")).otherwise(0.0)))
