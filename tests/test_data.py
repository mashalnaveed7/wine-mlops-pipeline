from src.data import load_wine_data


def test_data_split():
    X_train, X_test, y_train, y_test = load_wine_data()

    assert X_train.shape[1] == 13
    assert X_test.shape[1] == 13

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_train_test_size():
    X_train, X_test, _, _ = load_wine_data()

    total_samples = len(X_train) + len(X_test)

    assert total_samples == 178
    assert len(X_test) == 36
