"""Utilidades PySpark para la evaluación técnica. Uso:  from src.utils import *"""
from .spark_session import crear_spark
from .eda import (resumen_datos, distribucion_target, tipos_variables, detectar_columnas_problema,
                  tasa_target_por_categoria, tasa_target_por_presencia, graficar_numericas_vs_target,
                  matriz_correlacion, graficar_correlacion, pares_correlacionados, muestreo_estratificado)
from .outliers import limites_iqr, resumen_atipicos, eliminar_atipicos_iqr, crear_capping
from .feature_selection import (calcular_iv, tabla_iv, filtrar_por_iv, filtrar_correlacion, calcular_vif,
                                vif_iterativo, importancia_arbol, seleccionar_por_importancia)
from .weights import COL_PESO, agregar_pesos, pesos_por_clase
from .threshold import (COL_PROB, agregar_prob, buscar_umbral, umbral_youden, umbral_con_restriccion,
                        graficar_umbral)
from .metrics import (metricas_completas, metricas_confusion, metricas_curva, evaluar_modelo, calcular_gaps,
                      diagnostico_sobreajuste, graficar_roc, graficar_matriz_confusion)
from .modeling import (split_train_val_test, construir_pipeline, modelos_base, entrenar_y_evaluar,
                       tabla_comparativa, elegir_campeon, espacios_busqueda, agregar_folds, tunear_modelo)
from .mlflow_utils import (ALIAS, configurar_mlflow, loguear_run, registrar_campeon, cargar_campeon,
                           predecir_con_campeon, listar_runs, registrar_desde_run,
                           loguear_run_cluster, registrar_segmentador, cargar_segmentador)
from .clustering import (COL_CLUSTER, construir_preprocesamiento, elegir_n_pca, crear_modelo, silhouette, tamanos,
                         evaluar_k, elegir_k, challenge_algoritmos, elegir_campeon_cluster, estabilidad_semillas,
                         unir_pipeline, perfilar_clusters, perfil_relativo, rasgos_principales,
                         graficar_codo_silhouette, graficar_clusters_2d, graficar_perfil)
