"""Entrenamiento supervisado end-to-end con PySpark + Spark ML + MLflow.
Ejecutar desde la raíz:  python -m src.train
Cambia solo la sección CONFIG para el dataset de la prueba."""
import json

import matplotlib
matplotlib.use("Agg")

from src.features import TARGET, leer_datos, preparar_features, preparar_target
from src.utils import *

# ============================================================ CONFIG
RUTA_DATOS = "data/web_phishing.csv"
EXPERIMENTO = "prueba_spark"
NOMBRE_MODELO = "modelo_phishing"
RANDOM_STATE = 42
ELIMINAR_DUPLICADOS = False        # False: las filas repetidas son URLs distintas con los mismos conteos
TUNEAR_TOP = 2                     # cuántos modelos del challenge se tunean
FOLDS_TUNEO = 3
FRACCION_TUNEO = 0.3               # fracción de train para la búsqueda (1.0 = todo). El campeón se reentrena con todo train
# Atípicos (IQR 1.5). Cada variable numérica cae en uno de tres grupos:
ATIPICOS_ELIMINAR = []             # elimina las FILAS de train con atípicos en estas variables
ATIPICOS_CAPPING = ["url_length", "n_slash"]            # None = todas las demás numéricas | [] = ninguna | lista = solo esas
                                   # las numéricas que no estén en ninguna lista quedan sin tratamiento
GUARDAR_MODELOS_CHALLENGE = True   # guarda el modelo en cada run (permite registrarlo después)


def main():
    spark = crear_spark()

    # ---------------------------------------------------- 1. Datos
    df = preparar_target(preparar_features(leer_datos(spark, RUTA_DATOS)))
    if ELIMINAR_DUPLICADOS:
        df = df.dropDuplicates()

    # ---------------------------------------------------- 2. Split (antes de transformar)
    train, val, test = split_train_val_test(df, TARGET, seed=RANDOM_STATE)
    num_cols, cat_cols = tipos_variables(train, TARGET)

    # ---------------------------------------------------- 2b. Atípicos
    # Eliminar: quita filas SOLO de train (val/test no se tocan: en producción no se borran casos).
    # Capping: límites de train dentro del Pipeline (mismos en val/test/producción).
    eliminar = [c for c in ATIPICOS_ELIMINAR if c in num_cols]
    if ATIPICOS_CAPPING is None:
        capping = [c for c in num_cols if c not in eliminar]
    else:
        capping = [c for c in ATIPICOS_CAPPING if c in num_cols]
    repetidas = set(eliminar) & set(capping)
    if repetidas:
        raise ValueError(f"Variables en ELIMINAR y CAPPING a la vez: {sorted(repetidas)}")
    sin_tratar = [c for c in num_cols if c not in eliminar and c not in capping]
    print(f"Atípicos -> eliminar filas: {eliminar} | capping: {capping} | sin tratamiento: {sin_tratar}")
    if eliminar:
        train, resumen_elim = eliminar_atipicos_iqr(train, eliminar, TARGET)
        print(resumen_elim)
    limites_cap = limites_iqr(train, capping)
    sin_rango = [c for c, l in limites_cap.items() if l is None]
    if sin_rango:
        print(f"Capping: IQR=0, no se recortan: {sin_rango}")

    # ---------------------------------------------------- 3. Pesos (una vez, con train)
    train, _ = agregar_pesos(train, TARGET)
    train = train.cache()

    # ---------------------------------------------------- 4. Selección de variables (solo train)
    t_iv = tabla_iv(train, num_cols + cat_cols, TARGET)
    print(t_iv.to_string())
    pasan_iv, _ = filtrar_por_iv(t_iv)
    iv_dict = dict(zip(t_iv["variable"], t_iv["iv"]))

    # Set árboles: IV + importancia (GBT con pesos)
    imp = importancia_arbol(train, [c for c in pasan_iv if c in num_cols], [c for c in pasan_iv if c in cat_cols],
                            TARGET, weight_col=COL_PESO, seed=RANDOM_STATE)
    print(imp.round(4).to_string())
    feats_arboles = seleccionar_por_importancia(imp, acumulado=0.95)

    # Set logit: IV + correlación + VIF (sobre numéricas); categóricas que pasan IV se mantienen
    num_iv = [c for c in pasan_iv if c in num_cols]
    num_corr, _ = filtrar_correlacion(train, num_iv, umbral=0.8, prioridad=iv_dict)
    num_vif, _, _ = vif_iterativo(train, num_corr, umbral=5)
    feats_logit = num_vif + [c for c in pasan_iv if c in cat_cols]

    with open("features.json", "w") as f:
        json.dump({"arboles": feats_arboles, "logit": feats_logit}, f, indent=2)

    def split_cols(feats):
        return [c for c in feats if c in num_cols], [c for c in feats if c in cat_cols]

    # ---------------------------------------------------- 5. Challenge de modelos (con pesos)
    configurar_mlflow(EXPERIMENTO)
    data = {"train": train, "val": val, "test": test}
    ejemplo = val.select(*num_cols, *cat_cols)          # columnas de entrada del modelo
    base = modelos_base(TARGET, RANDOM_STATE)
    config = {  # nombre: (features, escalar)
        "logit": (feats_logit, True),
        "random_forest": (feats_arboles, False),
        "gbt": (feats_arboles, False),
    }

    resultados = {}
    for nombre, (feats, escalar) in config.items():
        print(f"\n--- Entrenando {nombre} ---")
        nc, cc = split_cols(feats)
        pipe = construir_pipeline(base[nombre], nc, cc, escalar=escalar, limites_capping=limites_cap)
        r = entrenar_y_evaluar(nombre, pipe, data, TARGET)
        r["features"] = feats
        p_val = agregar_prob(r["pipeline"].transform(val)).select(TARGET, COL_PROB).cache()
        figs = {"roc_val": graficar_roc({nombre: p_val}, TARGET),
                "umbral_val": graficar_umbral(p_val, TARGET, r["umbral"], titulo=nombre),
                "cm_val_opt": graficar_matriz_confusion(p_val, TARGET, r["umbral"])}
        p_val.unpersist()
        loguear_run(r, figs, etiqueta="challenge", sdf_ejemplo=ejemplo if GUARDAR_MODELOS_CHALLENGE else None)
        resultados[nombre] = r

    tabla = tabla_comparativa(resultados)
    print("\n=== CHALLENGE ===\n", tabla.round(4).to_string())

    # ---------------------------------------------------- 6. Tuneo de los mejores (CrossValidator)
    for nombre in tabla["modelo"].head(TUNEAR_TOP):
        print(f"\n--- Tuneando {nombre} ---")
        feats, escalar = config[nombre]
        nc, cc = split_cols(feats)
        modelo = modelos_base(TARGET, RANDOM_STATE)[nombre]   # instancia nueva para la grilla
        pipe = construir_pipeline(modelo, nc, cc, escalar=escalar, limites_capping=limites_cap)
        mejor, mejores_params, _ = tunear_modelo(pipe, espacios_busqueda(nombre, modelo), train, TARGET,
                                                 folds=FOLDS_TUNEO, seed=RANDOM_STATE, fraccion=FRACCION_TUNEO)
        r = entrenar_y_evaluar(f"{nombre}_tuned", mejor, data, TARGET)
        r["features"] = feats
        loguear_run(r, etiqueta="tuneo", params_extra={f"best_{k}": v for k, v in mejores_params.items()},
                    sdf_ejemplo=ejemplo if GUARDAR_MODELOS_CHALLENGE else None)
        resultados[f"{nombre}_tuned"] = r

    tabla = tabla_comparativa(resultados)
    print("\n=== CHALLENGE + TUNEO ===\n", tabla.round(4).to_string())
    tabla.to_csv("tabla_comparativa.csv", index=False)

    # ---------------------------------------------------- 7. Campeón: test se mira aquí
    # Para registrar otro modelo después: python -m src.registrar --run-id <id>
    campeon = elegir_campeon(tabla, metrica="auc_val", max_gap=0.05)
    r = resultados[campeon]
    print(f"\nCAMPEÓN: {campeon} | umbral={r['umbral']} | {r['diagnostico']}")
    print(r["metricas"].round(4).to_string())

    preds = {k: agregar_prob(r["pipeline"].transform(s)).select(TARGET, COL_PROB).cache() for k, s in data.items()}
    figs = {"roc_train_val_test": graficar_roc(preds, TARGET),
            "cm_test_opt": graficar_matriz_confusion(preds["test"], TARGET, r["umbral"], "Test - umbral óptimo"),
            "cm_test_05": graficar_matriz_confusion(preds["test"], TARGET, 0.5, "Test - umbral 0.5")}
    registrar_campeon(r, ejemplo, NOMBRE_MODELO, features=r["features"], figuras=figs)
    for p in preds.values():
        p.unpersist()
    spark.stop()
    return resultados, tabla


if __name__ == "__main__":
    main()
