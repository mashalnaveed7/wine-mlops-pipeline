from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_wine_data():
    """Load Wine dataset and split it into training and test sets."""

    wine = load_wine()

    X = wine.data
    y = wine.target

    validate_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def validate_data(X, y):
    """Validate the Wine dataset."""

    if X.shape[1] != 13:
        raise ValueError("Wine dataset must have exactly 13 features.")

    if X.shape[0] != len(y):
        raise ValueError("Number of samples and targets must match.")

    if hasattr(X, "any"):
        if not X.any():
            raise ValueError("Dataset is empty.")

    if hasattr(X, "shape"):
        if X is None or y is None:
            raise ValueError("Data cannot be None.")

    if hasattr(X, "dtype"):
        if X.dtype.kind in "fc":
            if not (X == X).all():
                raise ValueError("Features contain null values.")

    if hasattr(y, "dtype"):
        if y.dtype.kind in "fc":
            if not (y == y).all():
                raise ValueError("Targets contain null values.")
