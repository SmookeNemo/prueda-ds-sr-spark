"""Chequeo rápido del entorno: Java, SparkSession, Spark ML y guardado de modelos.
Uso:  python -m src.verificar_spark"""
import os
import shutil
import sys
import tempfile


def main():
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"JAVA_HOME: {os.environ.get('JAVA_HOME', '(no definido)')}")
    if os.name == "nt":
        print(f"HADOOP_HOME: {os.environ.get('HADOOP_HOME', '(no definido)')}")

    import pyspark
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import VectorAssembler

    from src.utils import crear_spark
    print(f"PySpark: {pyspark.__version__}")

    spark = crear_spark("verificacion")
    print(f"Spark {spark.version} OK")
    df = spark.createDataFrame([(0.0, 1.0, 0), (1.0, 0.0, 1), (0.5, 0.5, 1), (0.1, 0.9, 0)], ["a", "b", "y"])
    modelo = Pipeline(stages=[VectorAssembler(inputCols=["a", "b"], outputCol="features"),
                              LogisticRegression(labelCol="y")]).fit(df)
    print("Entrenamiento Spark ML OK")

    tmp = tempfile.mkdtemp()
    try:
        modelo.write().overwrite().save(os.path.join(tmp, "modelo"))
        print("Guardado de modelos OK (necesario para MLflow)")
    except Exception as e:
        print("ERROR al guardar el modelo. En Windows suele faltar winutils.exe / hadoop.dll "
              "(ver README, sección Windows).")
        print(f"Detalle: {str(e)[:300]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        spark.stop()


if __name__ == "__main__":
    main()
