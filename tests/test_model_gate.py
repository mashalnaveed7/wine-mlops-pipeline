import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.data import load_wine_data


def test_model_quality_gate():
    X_train, X_test, y_train, y_test = load_wine_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    validation_predictions = model.predict(X_test)

    validation_f1 = f1_score(
        y_test,
        validation_predictions,
        average="macro",
    )

    assert validation_f1 >= 0.88, (
        f"Validation Macro F1 {validation_f1:.4f} "
        "is below required threshold 0.88"
    )


def test_inference_latency_gate():
    X_train, X_test, y_train, y_test = load_wine_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    start_time = time.perf_counter()

    model.predict(X_test)

    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    assert latency_ms <= 30, (
        f"Inference latency {latency_ms:.2f} ms "
        "is above required limit of 30 ms"
    )


def test_output_schema():
    X_train, X_test, y_train, y_test = load_wine_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    allowed_classes = {0, 1, 2}

    assert set(predictions).issubset(allowed_classes), (
        f"Invalid output classes found: {set(predictions)}"
    )
