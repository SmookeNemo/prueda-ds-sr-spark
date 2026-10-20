"""Creación de la SparkSession local."""
import os
import sys

from pyspark.sql import SparkSession


def crear_spark(app: str = "prueba-ds-spark", memoria: str = "4g", particiones: int = 8) -> SparkSession:
    """SparkSession local. Fija el Python del entorno virtual para los workers
    (en Windows evita el error 'Python worker failed to connect back')."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (SparkSession.builder
             .appName(app)
             .master("local[*]")
             .config("spark.driver.memory", memoria)
             .config("spark.sql.shuffle.partitions", str(particiones))
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark
