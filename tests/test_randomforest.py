import pandas as pd
from chefboost import Chefboost as cb
from chefboost.commons.logger import Logger

logger = Logger(module="tests/test_randomforest.py")


def test_randomforest_for_classification():
    config = {
        "algorithm": "ID3",
        "enableRandomForest": True,
        "num_of_trees": 3,
    }
    df = pd.read_csv("dataset/car.data")

    model = cb.fit(df, config, silent=True)

    assert model["config"]["algorithm"] == "ID3"
    assert model["evaluation"]["train"]["Accuracy"] > 90

    # feature importance
    decision_rules = []
    for tree in model["trees"]:
        decision_rule = tree.__dict__["__spec__"].origin
        decision_rules.append(decision_rule)

    df = cb.feature_importance(decision_rules, silent=True)
    assert df.shape[0] == 6

    # this is not in train data
    instance = ["high", "high", 4, "more", "big", "high"]
    prediction = cb.predict(model, instance)
    assert prediction in ["unacc", "acc"]

    instance = ["vhigh", "vhigh", 2, "2", "small", "low"]
    prediction = cb.predict(model, instance)
    assert prediction in ["unacc", "acc"]


def test_randomforest_for_regression():
    config = {
        "algorithm": "ID3",
        "enableRandomForest": True,
        "num_of_trees": 5,
    }
    df = pd.read_csv("dataset/car_reg.data")
    model = cb.fit(df, config, silent=True)

    assert model["evaluation"]["train"]["MAE"] < 30
    assert model["config"]["algorithm"] == "Regression"

    instance = ["high", "high", 4, "more", "big", "high"]
    target = 100
    prediction = cb.predict(model, instance)
    assert abs(prediction - target) < 30
