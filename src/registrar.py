"""Registro manual de un modelo específico en el Model Registry.

Uso (desde la raíz del proyecto):
    python -m src.registrar                    # lista los runs con sus métricas
    python -m src.registrar --run-id <id>      # registra un run concreto como @champion
    python -m src.registrar --modelo logit     # registra el mejor run guardado de ese modelo
"""
import argparse

from src.utils import ALIAS, configurar_mlflow, listar_runs, registrar_desde_run

EXPERIMENTO = "prueba_spark"
NOMBRE_MODELO = "modelo_phishing"


def main():
    parser = argparse.ArgumentParser(description="Registrar un modelo específico en MLflow")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--run-id", help="run_id a registrar")
    grupo.add_argument("--modelo", help="nombre del modelo (tag 'modelo'), p. ej. logit o gbt_tuned")
    parser.add_argument("--alias", default=ALIAS, help="alias a asignar (por defecto: champion)")
    args = parser.parse_args()

    configurar_mlflow(EXPERIMENTO)
    runs = listar_runs(EXPERIMENTO)
    if not args.run_id and not args.modelo:
        print(runs.to_string())
        return

    run_id = args.run_id
    if args.modelo:
        cand = runs[(runs["modelo"] == args.modelo) & runs["tiene_modelo"] & (runs["etapa"] != "campeon")]
        if cand.empty:
            raise SystemExit(f"No hay runs con modelo guardado para '{args.modelo}'")
        run_id = cand.iloc[0]["run_id"]
    registrar_desde_run(run_id, NOMBRE_MODELO, alias=args.alias)


if __name__ == "__main__":
    main()
