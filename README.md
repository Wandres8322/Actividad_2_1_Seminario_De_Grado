
# Estandarización de Pipelines y Control de Calidad de Datos

## 1. Descripción

Este proyecto automatiza la validación de datos y el entrenamiento de un clasificador Random Forest sobre el conjunto Iris. Mantiene la continuidad experimental de la Actividad 1.1: cuatro configuraciones de `max_depth`, partición 80/20, `random_state=42` y registro de parámetros, métricas y modelos mediante MLflow.

## 2. Estructura

```text
Actividad_2_1_Pipeline/
├── src/
│   ├── __init__.py
│   ├── validate_data.py
│   └── train.py
├── logs/
├── models/
├── data_contract.json
├── requirements.txt
├── run_pipeline.py
├── README.md
└── .gitignore
```

## 3. Requisitos

- Python 3.12.
- Git (opcional, para clonar el repositorio).
- Conexión a internet durante la instalación de dependencias.

## 4. Instalación en Windows

Clonar el repositorio y entrar en la carpeta del proyecto:

```bash
git clone URL_DEL_REPOSITORIO
cd Actividad_2_1_Pipeline
```

Crear y activar un entorno virtual:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activación, ejecutar:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Ejecución completa

Desde la raíz del proyecto, ejecutar:

```bash
python run_pipeline.py
```

El orquestador ejecuta secuencialmente:

1. Validación de estructura, tipos, ausencia de nulos, rangos y clases del dataset.
2. Entrenamiento de las cuatro configuraciones Random Forest.
3. Registro de parámetros, métricas y modelos en MLflow.

Si la validación falla, el entrenamiento no se ejecuta. El proceso termina con un código de error y registra el resultado en `logs/pipeline.log`.

## 6. Validación independiente

Para ejecutar únicamente los controles de calidad:

```bash
python -m src.validate_data
```

La lógica de validación está implementada con la biblioteca estándar de Python. No utiliza Great Expectations ni otra librería externa de validación. Scikit-learn se utiliza para cargar el dataset Iris.

## 7. Contrato de datos

El archivo `data_contract.json` define las cuatro variables requeridas para una futura inferencia, su tipo, rango permitido, obligatoriedad y ejemplos válidos e inválidos.

Los rangos corresponden a los límites observados en el conjunto Iris utilizado en esta práctica. No deben interpretarse como límites biológicos universales para cualquier población de flores.

## 8. Seguimiento de experimentos

MLflow almacena localmente los metadatos en `mlflow.db` y los artefactos en `mlruns/`. Para consultar la interfaz:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir en el navegador: http://127.0.0.1:5000

## 9. Reproducibilidad

La ejecución conserva el dataset Iris de scikit-learn, `test_size=0.20`, `random_state=42`, `n_estimators=100` y las cuatro configuraciones de profundidad. Las versiones de las dependencias principales están fijadas en `requirements.txt`.

## 10. Archivos que deben versionarse

Versionar el código fuente, `data_contract.json`, `requirements.txt`, `README.md` y `.gitignore`. No subir `.venv/`, `mlflow.db`, `mlruns/` ni archivos de log generados localmente.

## 11. Alcance

La actividad implementa automatización, validación básica de calidad y contrato de datos. No incluye API de inferencia, Docker, despliegue ni monitoreo de producción.
