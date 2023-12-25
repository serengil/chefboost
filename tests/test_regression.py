import pandas as pd
from chefboost import Chefboost as cb
from chefboost.commons.logger import Logger

logger = Logger(module="tests/test_regression.py")


def test_c45_for_nominal_features_and_numeric_target():
    df = pd.read_csv("dataset/golf3.txt")
    _ = cb.fit(df, config={"algorithm": "Regression"}, silent=True)
    logger.info("✅ build regression for nominal features and numeric target test done")


def test_c45_for_nominal_and_numeric_features_and_numeric_target():
    df = pd.read_csv("dataset/golf4.txt")
    _ = cb.fit(df, config={"algorithm": "Regression"}, silent=True)
    logger.info(
        "✅ build regression tree for nominal and numeric features and numeric target test done"
    )


def test_switching_to_regression_tree():
    df = pd.read_csv("dataset/golf4.txt")
    config = {"algorithm": "ID3"}
    model = cb.fit(df, config, silent=True)
    assert model["config"]["algorithm"] == "Regression"
    logger.info("✅ switching to regression tree test done")
