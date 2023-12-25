import pandas as pd
from chefboost import Chefboost as cb
from chefboost.commons.logger import Logger

logger = Logger(module="tests/test_cart.py")


def test_cart_for_nominal_features_and_nominal_target():
    df = pd.read_csv("dataset/golf.txt")
    model = cb.fit(df, config={"algorithm": "CART"}, silent=True)
    assert model["config"]["algorithm"] == "CART"
    logger.info("✅ build cart for nominal and numeric features and nominal target test done")


def test_cart_for_nominal_and_numeric_features_and_nominal_target():
    df = pd.read_csv("dataset/golf2.txt")
    model = cb.fit(df, config={"algorithm": "CART"}, silent=True)
    assert model["config"]["algorithm"] == "CART"
    logger.info("✅ build cart for nominal and numeric features and nominal target test done")

def test_large_dataset():
    df = pd.read_csv("dataset/car.data")
    model = cb.fit(df, config={"algorithm": "CART"}, silent=True)
    assert model["config"]["algorithm"] == "CART"
    logger.info("✅ build c4.5 for large dataset test done")