import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from src.data import load_wine_data


EXPERIMENT_NAME = "Wine-Cultivar-Classification"
MODEL_NAME = "WineClassifier"


def calculate_metrics(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    f1 = f1_score(y, predictions, average="macro")
    accuracy = accuracy_score(y, predictions)
    loss = log_loss(y, probabilities)

    return f1, accuracy, loss


def cross_validate_model(model_class, parameters, X, y):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    train_results = []
    validation_results = []

    for train_index, val_index in cv.split(X, y):
        X_fold_train = X[train_index]
        X_fold_val = X[val_index]

        y_fold_train = y[train_index]
        y_fold_val = y[val_index]

        model = model_class(**parameters)

        model.fit(X_fold_train, y_fold_train)

        train_results.append(
            calculate_metrics(model, X_fold_train, y_fold_train)
        )

        validation_results.append(
            calculate_metrics(model, X_fold_val, y_fold_val)
        )

    train_f1 = sum(x[0] for x in train_results) / 5
    train_accuracy = sum(x[1] for x in train_results) / 5
    train_loss = sum(x[2] for x in train_results) / 5

    val_f1 = sum(x[0] for x in validation_results) / 5
    val_accuracy = sum(x[1] for x in validation_results) / 5
    val_loss = sum(x[2] for x in validation_results) / 5

    return (
        train_f1,
        train_accuracy,
        train_loss,
        val_f1,
        val_accuracy,
        val_loss,
    )


def run_experiments():
    X_train, _, y_train, _ = load_wine_data()

    mlflow.set_experiment(EXPERIMENT_NAME)

    configurations = {
        "RandomForest": (
            RandomForestClassifier,
            {
                "n_estimators": [50, 100, 150],
                "max_depth": [None],
                "random_state": [42],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier,
            {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.05],
                "random_state": [42],
            },
        ),
    }

    best_f1 = -1
    best_run_id = None

    for model_name, (model_class, grid) in configurations.items():
        for number, parameters in enumerate(
            ParameterGrid(grid),
            start=1,
        ):
            print(f"\n{model_name} - Configuration {number}")

            metrics = cross_validate_model(
                model_class,
                parameters,
                X_train,
                y_train,
            )

            (
                train_f1,
                train_accuracy,
                train_loss,
                val_f1,
                val_accuracy,
                val_loss,
            ) = metrics

            final_model = model_class(**parameters)
            final_model.fit(X_train, y_train)

            signature = infer_signature(
                X_train,
                final_model.predict(X_train),
            )

            input_example = X_train[:5]

            with mlflow.start_run(
                run_name=f"{model_name}_config_{number}"
            ) as run:
                mlflow.log_param("model_family", model_name)
                mlflow.log_params(parameters)

                mlflow.log_metric("train_macro_f1", train_f1)
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("train_log_loss", train_loss)

                mlflow.log_metric("validation_macro_f1", val_f1)
                mlflow.log_metric("validation_accuracy", val_accuracy)
                mlflow.log_metric("validation_log_loss", val_loss)

                mlflow.set_tag("model_family", model_name)
                mlflow.set_tag("purpose", "hyperparameter_tuning")

                mlflow.sklearn.log_model(
                    sk_model=final_model,
                    name="model",
                    signature=signature,
                    input_example=input_example,
                    skops_trusted_types=["sklearn.tree._tree.Tree"],
                )

                print(f"Validation Macro F1: {val_f1:.4f}")
                print(f"Validation Accuracy: {val_accuracy:.4f}")
                print(f"Validation Log Loss: {val_loss:.4f}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_run_id = run.info.run_id

    print("\nBest Validation Macro F1:", round(best_f1, 4))
    print("Best Run ID:", best_run_id)

    model_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=registered_model.version,
    )

    print("\nModel registered successfully.")
    print("Registered Model:", MODEL_NAME)
    print("Version:", registered_model.version)
    print("Alias: champion")


if __name__ == "__main__":
    run_experiments()
