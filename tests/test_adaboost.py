import pandas as pd
from chefboost import Chefboost as cb
from chefboost.commons.logger import Logger

logger = Logger(module="tests/test_adaboost.py")


def test_adaboost():
    config = {
        "algorithm": "Regression",
        "enableAdaboost": True,
        "num_of_weak_classifier": 10,
        "enableParallelism": False,
    }
    df = pd.read_csv("dataset/adaboost.txt")
    validation_df = df.copy()

    model = cb.fit(df, config, validation_df=validation_df, silent=True)

    instance = [4, 3.5]

    prediction = cb.predict(model, instance)

    assert prediction == -1
    assert len(model["trees"]) > 1

    logger.info("✅ adaboost model restoration test done")
