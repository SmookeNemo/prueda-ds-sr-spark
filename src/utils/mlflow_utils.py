"""MLflow con modelos de Spark ML: tracking de todos los modelos del challenge y registro del campeón.

- Cada modelo evaluado se loguea como un run (params, métricas, umbral, gráficos y el PipelineModel),
  pero NO se registra.
- El campeón se registra en el Model Registry con alias 'champion' y el umbral como tag.
- Registro manual: registrar_desde_run(run_id) registra cualquier run guardado.
- predict.py carga 'models:/<nombre>@champion' (un PipelineModel de Spark) y lee el umbral del tag.
"""
import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import mlflow.spark
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .threshold import COL_PROB, agregar_prob

ALIAS = "champion"


def configurar_mlflow(experimento: str, tracking_uri: str = "sqlite:///mlflow.db"):
    """SQLite permite usar el Model Registry en local. UI: mlflow ui --backend-store-uri sqlite:///mlflow.db"""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experimento)
    print(f"MLflow -> {tracking_uri} | experimento: {experimento}")


def _params_simples(pipeline_model) -> dict:
    """Hiperparámetros del modelo (último stage del PipelineModel)."""
    modelo = pipeline_model.stages[-1]
    return {p.name: v for p, v in modelo.extractParamMap().items()
            if isinstance(v, (int, float, str, bool)) and v is not None}


def _metricas_planas(df_eval: pd.DataFrame) -> dict:
    """val_opt_auc, test_0.5_f1, ... (una métrica por conjunto x umbral x métrica)."""
    out = {}
    excluir = {"modelo", "conjunto", "tipo_umbral", "umbral"}
    for _, fila in df_eval.iterrows():
        pref = f"{fila['conjunto']}_{'opt' if fila['tipo_umbral'] == 'optimo' else '0.5'}"
        for col, val in fila.items():
            if col not in excluir:
                out[f"{pref}_{col}"] = float(val)
    return out


def _log_model(pipeline_model, sdf_ejemplo: DataFrame, registered_model_name: str = None):
    """Guarda el PipelineModel de Spark en el run activo. La firma describe las columnas de entrada."""
    firma = infer_signature(sdf_ejemplo.limit(5).toPandas())
    kwargs = dict(spark_model=pipeline_model, signature=firma, registered_model_name=registered_model_name)
    try:
        return mlflow.spark.log_model(name="model", **kwargs)          # MLflow 3.x
    except TypeError:
        return mlflow.spark.log_model(artifact_path="model", **kwargs)  # MLflow 2.x


def _etiquetar_version(nombre_modelo: str, version, umbral: float, algoritmo: str,
                       features: list = None, alias: str = ALIAS):
    client = MlflowClient()
    client.set_registered_model_alias(nombre_modelo, alias, version)
    client.set_model_version_tag(nombre_modelo, version, "umbral", str(umbral))
    client.set_model_version_tag(nombre_modelo, version, "algoritmo", algoritmo)
    if features:
        client.set_model_version_tag(nombre_modelo, version, "features", ",".join(features))
    print(f"Registrado '{nombre_modelo}' v{version} ({algoritmo}) con alias '@{alias}' | umbral={umbral}")


def _log_comun(resultado: dict, figuras: dict, params_extra: dict = None):
    mlflow.set_tag("modelo", resultado["nombre"])
    mlflow.log_params(_params_simples(resultado["pipeline"]))
    if params_extra:
        mlflow.log_params(params_extra)
    mlflow.log_param("umbral_optimo", resultado["umbral"])
    mlflow.log_metric("umbral_youden", resultado["umbral_youden"])
    mlflow.log_metrics(_metricas_planas(resultado["metricas"]))
    mlflow.log_metrics({k: float(v) for k, v in resultado["gaps"].items()})
    mlflow.set_tag("diagnostico", resultado["diagnostico"])
    if resultado.get("features"):
        mlflow.set_tag("features", ",".join(resultado["features"]))
    for nombre, fig in (figuras or {}).items():
        mlflow.log_figure(fig, f"graficos/{nombre}.png")
        plt.close(fig)
    with tempfile.TemporaryDirectory() as tmp:
        ruta = os.path.join(tmp, f"metricas_{resultado['nombre']}.csv")
        resultado["metricas"].to_csv(ruta, index=False)
        mlflow.log_artifact(ruta, artifact_path="tablas")


def loguear_run(resultado: dict, figuras: dict = None, etiqueta: str = "challenge", params_extra: dict = None,
                sdf_ejemplo: DataFrame = None):
    """Loguea un modelo SIN registrarlo. Si se pasa sdf_ejemplo, guarda también el PipelineModel
    en el run, para poder registrarlo después a mano."""
    with mlflow.start_run(run_name=f"{etiqueta}_{resultado['nombre']}") as run:
        mlflow.set_tag("etapa", etiqueta)
        _log_comun(resultado, figuras, params_extra)
        if sdf_ejemplo is not None:
            _log_model(resultado["pipeline"], sdf_ejemplo)   # guardado, NO registrado
        return run.info.run_id


def registrar_campeon(resultado: dict, sdf_ejemplo: DataFrame, nombre_modelo: str,
                      features: list = None, figuras: dict = None) -> int:
    """Loguea y REGISTRA el campeón, le asigna el alias 'champion' y guarda el umbral como tag."""
    with mlflow.start_run(run_name=f"campeon_{resultado['nombre']}"):
        mlflow.set_tag("etapa", "campeon")
        _log_comun(resultado, figuras)
        info = _log_model(resultado["pipeline"], sdf_ejemplo, registered_model_name=nombre_modelo)
    version = info.registered_model_version
    _etiquetar_version(nombre_modelo, version, resultado["umbral"], resultado["nombre"], features)
    return int(version)


# ----------------------------------------------------------------- Registro manual
def _runs_con_modelo(experiment_id: str) -> set:
    """run_ids con modelo guardado. Revisa las dos formas en que MLflow guarda modelos:
    LoggedModels (MLflow 3, p. ej. sklearn) y la carpeta 'model' en los artefactos del run
    (es como lo guarda el flavor de Spark)."""
    ids = set()
    try:
        modelos = mlflow.search_logged_models(experiment_ids=[experiment_id], output_format="list")
        ids |= {m.source_run_id for m in modelos if m.source_run_id}
    except AttributeError:  # MLflow 2.x
        pass
    client = MlflowClient()
    for r in client.search_runs([experiment_id]):
        if r.info.run_id not in ids and any(a.path == "model" for a in client.list_artifacts(r.info.run_id)):
            ids.add(r.info.run_id)
    return ids


def listar_runs(experimento: str) -> pd.DataFrame:
    """Runs del experimento con las métricas clave, para elegir cuál registrar."""
    runs = mlflow.search_runs(experiment_names=[experimento], order_by=["metrics.val_opt_auc DESC"])
    if runs.empty:
        print("No hay runs en el experimento")
        return runs
    cols = {"run_id": "run_id", "tags.etapa": "etapa", "tags.modelo": "modelo",
            "params.umbral_optimo": "umbral", "metrics.val_opt_auc": "auc_val",
            "metrics.val_opt_f1": "f1_val_opt", "metrics.test_opt_auc": "auc_test",
            "metrics.gap_auc_train_val": "gap_auc_train_val"}
    tabla = runs[[c for c in cols if c in runs.columns]].rename(columns=cols)
    tabla.insert(1, "tiene_modelo", tabla["run_id"].isin(_runs_con_modelo(runs["experiment_id"].iloc[0])))
    return tabla.round(4).reset_index(drop=True)


def registrar_desde_run(run_id: str, nombre_modelo: str, alias: str = ALIAS) -> int:
    """Registra el modelo guardado en un run específico y le asigna el alias.
    El umbral y las features se toman del propio run."""
    run = MlflowClient().get_run(run_id)
    umbral = float(run.data.params.get("umbral_optimo", 0.5))
    algoritmo = run.data.tags.get("modelo", "desconocido")
    features = run.data.tags.get("features")
    try:
        mv = mlflow.register_model(f"runs:/{run_id}/model", nombre_modelo)
    except Exception as e:
        raise RuntimeError(f"El run {run_id} no tiene modelo guardado. Detalle: {e}")
    _etiquetar_version(nombre_modelo, mv.version, umbral, algoritmo,
                       features.split(",") if features else None, alias)
    return int(mv.version)


# ----------------------------------------------------------------- Carga y predicción
def cargar_campeon(nombre_modelo: str, alias: str = ALIAS):
    """Devuelve (PipelineModel de Spark, umbral, features o None). Requiere una SparkSession activa."""
    mv = MlflowClient().get_model_version_by_alias(nombre_modelo, alias)
    modelo = mlflow.spark.load_model(f"models:/{nombre_modelo}@{alias}")
    umbral = float(mv.tags.get("umbral", 0.5))
    features = mv.tags.get("features")
    print(f"Cargado '{nombre_modelo}' v{mv.version} (@{alias}) | umbral={umbral}")
    return modelo, umbral, features.split(",") if features else None


def predecir_con_campeon(sdf: DataFrame, nombre_modelo: str, alias: str = ALIAS) -> DataFrame:
    """Flujo de ejecución: carga el campeón, predice la probabilidad y aplica el umbral óptimo.
    Devuelve las columnas originales + probabilidad, prediccion y umbral_usado."""
    modelo, umbral, _ = cargar_campeon(nombre_modelo, alias)
    pred = agregar_prob(modelo.transform(sdf))
    return pred.select(*sdf.columns,
                       F.col(COL_PROB).alias("probabilidad"),
                       (F.col(COL_PROB) >= umbral).cast("int").alias("prediccion"),
                       F.lit(umbral).alias("umbral_usado"))


# ----------------------------------------------------------------- Segmentación (no supervisado)
def loguear_run_cluster(nombre_run: str, etapa: str, params: dict = None, metricas: dict = None,
                        figuras: dict = None, tablas: dict = None, modelo=None, sdf_ejemplo: DataFrame = None,
                        registered_model_name: str = None):
    """Run genérico para clustering: params, métricas, gráficos, tablas (CSV) y, opcionalmente, el modelo.
    Devuelve (run_id, info_del_modelo o None)."""
    info = None
    with mlflow.start_run(run_name=nombre_run) as run:
        mlflow.set_tag("etapa", etapa)
        if params:
            mlflow.log_params(params)
        if metricas:
            mlflow.log_metrics({k: float(v) for k, v in metricas.items()})
        for nombre, fig in (figuras or {}).items():
            mlflow.log_figure(fig, f"graficos/{nombre}.png")
            plt.close(fig)
        with tempfile.TemporaryDirectory() as tmp:
            for nombre, tabla in (tablas or {}).items():
                ruta = os.path.join(tmp, f"{nombre}.csv")
                tabla.to_csv(ruta)
                mlflow.log_artifact(ruta, artifact_path="tablas")
        if modelo is not None:
            info = _log_model(modelo, sdf_ejemplo, registered_model_name=registered_model_name)
        return run.info.run_id, info


def registrar_segmentador(modelo, sdf_ejemplo: DataFrame, nombre_modelo: str, algoritmo: str, k: int,
                          params: dict = None, metricas: dict = None, figuras: dict = None,
                          tablas: dict = None, alias: str = ALIAS) -> int:
    """Loguea y REGISTRA el modelo de segmentación campeón con alias y tags (algoritmo, k)."""
    _, info = loguear_run_cluster(f"campeon_{algoritmo}_k{k}", "campeon", params, metricas, figuras, tablas,
                                  modelo=modelo, sdf_ejemplo=sdf_ejemplo, registered_model_name=nombre_modelo)
    version = info.registered_model_version
    client = MlflowClient()
    client.set_registered_model_alias(nombre_modelo, alias, version)
    client.set_model_version_tag(nombre_modelo, version, "algoritmo", algoritmo)
    client.set_model_version_tag(nombre_modelo, version, "k", str(k))
    print(f"Registrado '{nombre_modelo}' v{version} ({algoritmo}, k={k}) con alias '@{alias}'")
    return int(version)


def cargar_segmentador(nombre_modelo: str, alias: str = ALIAS):
    """Devuelve (PipelineModel, k, algoritmo). Requiere una SparkSession activa."""
    mv = MlflowClient().get_model_version_by_alias(nombre_modelo, alias)
    modelo = mlflow.spark.load_model(f"models:/{nombre_modelo}@{alias}")
    k, alg = mv.tags.get("k"), mv.tags.get("algoritmo")
    print(f"Cargado '{nombre_modelo}' v{mv.version} (@{alias}) | {alg}, k={k}")
    return modelo, k, alg
