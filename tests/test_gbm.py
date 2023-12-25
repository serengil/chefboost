import pandas as pd
from chefboost import Chefboost as cb
from chefboost.commons.logger import Logger

logger = Logger(module="tests/test_gbm.py")


def test_gbm_regression():
    config = {
        "algorithm": "Regression",
        "enableGBM": True,
        "epochs": 10,
        "learning_rate": 1,
    }

    df = pd.read_csv("dataset/golf4.txt")
    validation_df = pd.read_csv("dataset/golf4.txt")

    model = cb.fit(df, config, validation_df=validation_df, silent=True)
    assert model["config"]["algorithm"] == "Regression"
    assert len(model["trees"]) > 1

    features = ["Sunny", 85, 85, "Weak"]
    target = 25
    prediction = cb.predict(model, features)
    assert abs(prediction - target) < 1


def test_gbm_classification():
    config = {
        "algorithm": "ID3",
        "enableGBM": True,
        "epochs": 10,
        "learning_rate": 1,
    }

    df = pd.read_csv(
        "dataset/iris.data",
        names=["Sepal length", "Sepal width", "Petal length", "Petal width", "Decision"],
    )
    validation_df = df.copy()

    model = cb.fit(df, config, validation_df=validation_df, silent=True)

    instance = [7.0, 3.2, 4.7, 1.4]
    target = "Iris-versicolor"
    prediction = cb.predict(model, instance)
    assert prediction == target
