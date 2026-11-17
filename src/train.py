
"""Entrenamiento y registro de cuatro configuraciones Random Forest en MLflow."""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 100
EXPERIMENT_NAME = "Proyecto_Final_Grado"

EXPERIMENTS = [
    {"run_name": "Comparativo_Baseline", "max_depth": None},
    {"run_name": "Comparativo_MaxDepth_1", "max_depth": 1},
    {"run_name": "Comparativo_MaxDepth_2", "max_depth": 2},
    {"run_name": "Comparativo_MaxDepth_3", "max_depth": 3},
]


def main():
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    for config in EXPERIMENTS:
        with mlflow.start_run(run_name=config["run_name"]):
            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=config["max_depth"],
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)

            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)

            mlflow.log_params({
                "n_estimators": N_ESTIMATORS,
                "max_depth": str(config["max_depth"]),
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "dataset": "Iris",
                "model": "RandomForestClassifier",
            })
            mlflow.log_metrics({
                "train_accuracy": train_accuracy,
                "accuracy": test_accuracy,
                "generalization_gap": train_accuracy - test_accuracy,
            })
            mlflow.sklearn.log_model(
                model,
                name="iris_model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            print(
                f"{config['run_name']}: "
                f"train_accuracy={train_accuracy:.4f} | "
                f"test_accuracy={test_accuracy:.4f} | "
                f"run_id={mlflow.active_run().info.run_id}"
            )

    print("\nEntrenamiento y registro en MLflow finalizados.")


if __name__ == "__main__":
    main()
