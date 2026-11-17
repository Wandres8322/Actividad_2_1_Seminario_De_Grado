
"""Validación de calidad para el conjunto Iris y entradas de inferencia.

La lógica de validación utiliza exclusivamente la biblioteca estándar de Python.
scikit-learn se utiliza únicamente para cargar el dataset Iris de referencia.
"""
from math import isfinite
from numbers import Real
from sklearn.datasets import load_iris

FEATURE_RANGES = {
    "sepal length (cm)": (4.3, 7.9),
    "sepal width (cm)": (2.0, 4.4),
    "petal length (cm)": (1.0, 6.9),
    "petal width (cm)": (0.1, 2.5),
}
EXPECTED_FEATURES = tuple(FEATURE_RANGES.keys())
VALID_TARGETS = {0, 1, 2}


def validate_record(record):
    """Valida un registro individual; devuelve una lista de errores."""
    errors = []

    if not isinstance(record, dict):
        return ["La entrada debe ser un objeto JSON."]

    missing = [name for name in EXPECTED_FEATURES if name not in record]
    extra = [name for name in record if name not in EXPECTED_FEATURES]

    if missing:
        errors.append(f"Faltan variables obligatorias: {missing}.")
    if extra:
        errors.append(f"Se encontraron variables no permitidas: {extra}.")

    for name in EXPECTED_FEATURES:
        if name not in record:
            continue

        value = record[name]
        if value is None:
            errors.append(f"'{name}' no puede ser nulo.")
            continue

        if isinstance(value, bool) or not isinstance(value, Real):
            errors.append(f"'{name}' debe ser numérico; se recibió {type(value).__name__}.")
            continue

        if not isfinite(float(value)):
            errors.append(f"'{name}' debe ser un número finito.")
            continue

        minimum, maximum = FEATURE_RANGES[name]
        if not minimum <= float(value) <= maximum:
            errors.append(
                f"'{name}' fuera de rango: {value}. "
                f"Rango permitido: [{minimum}, {maximum}] cm."
            )

    return errors


def validate_dataset():
    """Comprueba estructura, tipos, nulos, rangos y etiquetas del dataset Iris."""
    dataset = load_iris()
    errors = []

    if tuple(dataset.feature_names) != EXPECTED_FEATURES:
        errors.append("La estructura de variables del dataset no coincide con el contrato.")

    if dataset.data.shape[1] != len(EXPECTED_FEATURES):
        errors.append("Cantidad de variables predictoras incorrecta.")

    if len(dataset.data) != len(dataset.target):
        errors.append("La cantidad de observaciones y etiquetas no coincide.")

    for index, row in enumerate(dataset.data):
        record = dict(zip(EXPECTED_FEATURES, row.tolist()))
        row_errors = validate_record(record)
        if row_errors:
            errors.extend([f"Registro {index}: {message}" for message in row_errors])

    invalid_targets = sorted(set(dataset.target.tolist()) - VALID_TARGETS)
    if invalid_targets:
        errors.append(f"Etiquetas objetivo no permitidas: {invalid_targets}.")

    if errors:
        raise ValueError("Falló la validación de datos:\n- " + "\n- ".join(errors))

    return {
        "rows": int(dataset.data.shape[0]),
        "features": int(dataset.data.shape[1]),
        "classes": sorted(int(value) for value in set(dataset.target.tolist())),
    }


if __name__ == "__main__":
    summary = validate_dataset()
    print("VALIDACIÓN EXITOSA")
    print(f"Observaciones: {summary['rows']}")
    print(f"Variables predictoras: {summary['features']}")
    print(f"Clases objetivo: {summary['classes']}")
