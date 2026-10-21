"""Flujo de ejecución: asigna el segmento a clientes nuevos con el modelo de segmentación campeón.
Uso:  python -m src.segmentar data/bank_customer_segmentation.csv outputs/segmentos.csv"""
import os
import sys

from src.features_segmentacion import ID, leer_datos, preparar_features
from src.utils import COL_CLUSTER, cargar_segmentador, configurar_mlflow, crear_spark

NOMBRE_MODELO = "modelo_segmentacion"
EXPERIMENTO = "prueba_segmentacion"


def main(entrada: str, salida: str):
    spark = crear_spark("segmentar")
    configurar_mlflow(EXPERIMENTO)
    sdf = preparar_features(leer_datos(spark, entrada))
    modelo, k, alg = cargar_segmentador(NOMBRE_MODELO)
    pdf = modelo.transform(sdf).select(ID, COL_CLUSTER).toPandas()
    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    pdf.to_csv(salida, index=False)
    print(f"{len(pdf)} clientes segmentados en {salida}")
    print((pdf[COL_CLUSTER].value_counts(normalize=True).sort_index() * 100).round(1).rename("% clientes"))
    spark.stop()


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else "data/bank_customer_segmentation.csv"
    salida = sys.argv[2] if len(sys.argv) > 2 else "outputs/segmentos.csv"
    main(entrada, salida)
