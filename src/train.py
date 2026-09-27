import mlflow
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from src.data import load_wine_data


def calculate_metrics(model, X, y):
    """Calculate classification metrics."""

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    f1 = f1_score(y, predictions, average="macro")
    accuracy = accuracy_score(y, predictions)
    loss = log_loss(y, probabilities)

    return f1, accuracy, loss


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Train a model and calculate train and validation metrics."""

    model.fit(X_train, y_train)

    train_f1, train_accuracy, train_loss = calculate_metrics(
        model, X_train, y_train
    )

    val_f1, val_accuracy, val_loss = calculate_metrics(
        model, X_val, y_val
    )

    return (
        train_f1,
        train_accuracy,
        train_loss,
        val_f1,
        val_accuracy,
        val_loss,
    )


def cross_validate_model(model_class, parameters, X, y):
    """Perform 5-fold stratified cross-validation."""

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    train_metrics = []
    validation_metrics = []

    for train_index, val_index in cv.split(X, y):
        X_train = X[train_index]
        X_val = X[val_index]

        y_train = y[train_index]
        y_val = y[val_index]

        model = model_class(**parameters)

        metrics = evaluate_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        train_metrics.append(metrics[:3])
        validation_metrics.append(metrics[3:])

    train_f1 = sum(m[0] for m in train_metrics) / len(train_metrics)
    train_accuracy = sum(m[1] for m in train_metrics) / len(train_metrics)
    train_loss = sum(m[2] for m in train_metrics) / len(train_metrics)

    val_f1 = sum(m[0] for m in validation_metrics) / len(validation_metrics)
    val_accuracy = sum(m[1] for m in validation_metrics) / len(validation_metrics)
    val_loss = sum(m[2] for m in validation_metrics) / len(validation_metrics)

    return (
        train_f1,
        train_accuracy,
        train_loss,
        val_f1,
        val_accuracy,
        val_loss,
    )


def run_experiments():
    """Run hyperparameter experiments for both model families."""

    X_train, _, y_train, _ = load_wine_data()

    mlflow.set_experiment("Wine-Cultivar-Classification")

    model_configs = {
        "RandomForest": (
            RandomForestClassifier,
            {
                "n_estimators": [50, 100, 150],
                "max_depth": [None],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier,
            {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.05],
            },
        ),
    }

    for model_name, (model_class, grid) in model_configs.items():
        configurations = list(ParameterGrid(grid))

        for number, parameters in enumerate(configurations, start=1):
            print(f"\n{model_name} - Configuration {number}")
            print(parameters)

            with mlflow.start_run(
                run_name=f"{model_name}_config_{number}"
            ):
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

                mlflow.log_param("model_family", model_name)

                for name, value in parameters.items():
                    mlflow.log_param(name, value)

                mlflow.log_metric("train_macro_f1", train_f1)
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("train_log_loss", train_loss)

                mlflow.log_metric("validation_macro_f1", val_f1)
                mlflow.log_metric("validation_accuracy", val_accuracy)
                mlflow.log_metric("validation_log_loss", val_loss)

                print(f"Train Macro F1: {train_f1:.4f}")
                print(f"Train Accuracy: {train_accuracy:.4f}")
                print(f"Train Log Loss: {train_loss:.4f}")
                print(f"Validation Macro F1: {val_f1:.4f}")
                print(f"Validation Accuracy: {val_accuracy:.4f}")
                print(f"Validation Log Loss: {val_loss:.4f}")


if __name__ == "__main__":
    run_experiments()
