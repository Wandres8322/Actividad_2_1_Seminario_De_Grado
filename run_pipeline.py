
"""Orquesta validación de datos y entrenamiento con una sola ejecución."""
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")


def run_step(description, module):
    logger.info("INICIO: %s", description)
    subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT,
        check=True,
    )
    logger.info("COMPLETADO: %s", description)


def main():
    logger.info("Iniciando pipeline de clasificación Iris.")
    try:
        run_step("Validación de calidad de datos", "src.validate_data")
        run_step("Entrenamiento y registro de modelos", "src.train")
    except subprocess.CalledProcessError as exc:
        logger.exception("El pipeline se detuvo. Código de salida: %s", exc.returncode)
        raise SystemExit(exc.returncode) from exc

    logger.info("PIPELINE FINALIZADO CORRECTAMENTE.")


if __name__ == "__main__":
    main()
