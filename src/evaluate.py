import mlflow
from sklearn.metrics import accuracy_score, f1_score

from src.data import load_wine_data


MODEL_URI = "models:/WineClassifier@champion"


def evaluate_champion():
    _, X_test, _, y_test = load_wine_data()

    model = mlflow.pyfunc.load_model(MODEL_URI)

    predictions = model.predict(X_test)

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("\nChampion Model Test Results")
    print("---------------------------")
    print(f"Macro F1: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    evaluate_champion()
