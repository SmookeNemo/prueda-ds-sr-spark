"""Segmentación de clientes (no supervisado) con PySpark + Spark ML + MLflow.
Ejecutar desde la raíz:  python -m src.clustering
Cambia solo la sección CONFIG."""
import matplotlib
matplotlib.use("Agg")
import pandas as pd

from src.features_segmentacion import COLS_LOG, DERIVADAS, ID, VARIABLES, leer_datos, preparar_features
from src.utils import *

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

# ============================================================ CONFIG
RUTA_DATOS = "data/bank_customer_segmentation.csv"
EXPERIMENTO = "prueba_segmentacion"
NOMBRE_MODELO = "modelo_segmentacion"
RANDOM_STATE = 42
VAL_FRACCION = 0.2                 # validation para medir silhouette en datos no vistos
FEATURES = [v for v in VARIABLES if v != "TENURE"] + DERIVADAS   # variables que entran al clustering
VARIABLES_PERFIL = VARIABLES + DERIVADAS                          # variables para describir los segmentos
ATIPICOS_CAPPING = []              # el log1p ya controla las colas largas; [] = sin capping
USAR_PCA = True                    # reduce la redundancia (muchas variables correlacionadas)
VARIANZA_PCA = 0.85
K_RANGO = range(2, 11)
K_MIN_NEGOCIO = 3                  # k = 2 es demasiado grueso para segmentar clientes
MIN_PCT_CLUSTER = 0.03             # descarta soluciones con clusters de menos del 3 % de clientes
K_MANUAL = None                    # None = elección automática; o un número para forzar k
ALGORITMOS = ("kmeans", "bisecting_kmeans", "gmm")


def main():
    spark = crear_spark("segmentacion")

    # ---------------------------------------------------- 1. Datos y split
    sdf = preparar_features(leer_datos(spark, RUTA_DATOS)).cache()
    train, val = sdf.randomSplit([1 - VAL_FRACCION, VAL_FRACCION], seed=RANDOM_STATE)
    train, val = train.cache(), val.cache()
    print(f"Clientes: {sdf.count()} | train: {train.count()} | validation: {val.count()}")
    print(f"Variables para el clustering ({len(FEATURES)}): {FEATURES}")

    # ---------------------------------------------------- 2. Preprocesamiento (ajustado solo con train)
    limites = limites_iqr(train, ATIPICOS_CAPPING) if ATIPICOS_CAPPING else {}
    figs_prep = {}
    n_pca = None
    if USAR_PCA:
        n_pca, _, figs_prep["pca_varianza"] = elegir_n_pca(train, FEATURES, COLS_LOG, limites, VARIANZA_PCA)
    prep = construir_preprocesamiento(FEATURES, COLS_LOG, limites, n_pca).fit(train)
    train_t, val_t = prep.transform(train).cache(), prep.transform(val).cache()

    # ---------------------------------------------------- 3. Elección de k (codo + silhouette)
    configurar_mlflow(EXPERIMENTO)
    tabla_k = evaluar_k(train_t, val_t, K_RANGO, "kmeans", RANDOM_STATE)
    k = K_MANUAL or elegir_k(tabla_k, K_MIN_NEGOCIO, MIN_PCT_CLUSTER)
    print("\n=== ELECCIÓN DE k ===\n", tabla_k.round(4).to_string(index=False))
    tabla_k.to_csv("tabla_k.csv", index=False)
    loguear_run_cluster("seleccion_k", "seleccion_k",
                        params={"n_pca": n_pca, "k_elegido": k, "features": len(FEATURES)},
                        figuras={**figs_prep, "codo_silhouette": graficar_codo_silhouette(tabla_k, k)},
                        tablas={"tabla_k": tabla_k})

    # ---------------------------------------------------- 4. Challenge de algoritmos con el k elegido
    tabla_alg, modelos = challenge_algoritmos(train_t, val_t, k, ALGORITMOS, RANDOM_STATE)
    print("\n=== CHALLENGE ===\n", tabla_alg.round(4).to_string(index=False))
    tabla_alg.to_csv("tabla_challenge_clustering.csv", index=False)
    for _, fila in tabla_alg.iterrows():
        loguear_run_cluster(f"challenge_{fila['algoritmo']}_k{k}", "challenge",
                            params={"algoritmo": fila["algoritmo"], "k": k, "n_pca": n_pca},
                            metricas={c: fila[c] for c in ["silhouette_train", "silhouette_val", "gap_silhouette",
                                                           "pct_cluster_min", "pct_cluster_max"]})

    # ---------------------------------------------------- 5. Campeón y estabilidad
    campeon = elegir_campeon_cluster(tabla_alg, MIN_PCT_CLUSTER)
    print(f"\nCAMPEÓN: {campeon} con k={k}")
    estab = estabilidad_semillas(train_t, campeon, k)
    modelo_final = unir_pipeline(prep, modelos[campeon])

    # ---------------------------------------------------- 6. Perfilado de segmentos (todos los clientes)
    pred = modelo_final.transform(sdf).cache()
    perfil = perfilar_clusters(pred, VARIABLES_PERFIL)
    rel = perfil_relativo(pred, VARIABLES_PERFIL)
    rasgos = rasgos_principales(rel)
    print("\n=== PERFIL (medias por cluster) ===\n", perfil.round(2).to_string())
    print("\n=== ÍNDICE vs PROMEDIO GLOBAL ===\n", rel.to_string())
    print("\n=== RASGOS PRINCIPALES ===\n", rasgos.to_string())
    perfil.round(4).to_csv("perfil_clusters.csv")

    fila = tabla_alg[tabla_alg["algoritmo"] == campeon].iloc[0]
    registrar_segmentador(
        modelo_final, sdf.limit(5), NOMBRE_MODELO, campeon, k,
        params={"algoritmo": campeon, "k": k, "n_pca": n_pca, "capping": ",".join(ATIPICOS_CAPPING) or "ninguno"},
        metricas={"silhouette_train": fila["silhouette_train"], "silhouette_val": fila["silhouette_val"],
                  "ari_medio_semillas": estab["ari"].mean(), "pct_cluster_min": fila["pct_cluster_min"]},
        figuras={"clusters_2d": graficar_clusters_2d(pred), "perfil": graficar_perfil(rel)},
        tablas={"perfil": perfil, "perfil_relativo": rel, "rasgos": rasgos, "estabilidad": estab})
    spark.stop()


if __name__ == "__main__":
    main()
