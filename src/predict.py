"""Flujo de ejecución con el modelo campeón (PySpark).
Uso:  python -m src.predict data/nuevos.csv outputs/predicciones.csv"""
import os
import sys

from src.features import TARGET, leer_datos, preparar_features
from src.utils import configurar_mlflow, crear_spark, predecir_con_campeon

NOMBRE_MODELO = "modelo_phishing"
EXPERIMENTO = "prueba_spark"


def main(entrada: str, salida: str):
    spark = crear_spark("prediccion")
    configurar_mlflow(EXPERIMENTO)
    sdf = leer_datos(spark, entrada)
    sdf = preparar_features(sdf.drop(TARGET) if TARGET in sdf.columns else sdf)
    pred = predecir_con_campeon(sdf, NOMBRE_MODELO)
    # En local se guarda con pandas (evita depender de Hadoop/winutils en Windows).
    # En un clúster se usaría: pred.write.mode("overwrite").parquet(salida)
    pdf = pred.toPandas()
    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    pdf.to_csv(salida, index=False)
    print(f"{len(pdf)} predicciones guardadas en {salida} | tasa predicha: {pdf['prediccion'].mean():.4f}")
    spark.stop()


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else "data/nuevos.csv"
    salida = sys.argv[2] if len(sys.argv) > 2 else "outputs/predicciones.csv"
    main(entrada, salida)
