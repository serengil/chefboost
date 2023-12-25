import pandas as pd
from chefboost import Chefboost as cb
from chefboost.commons.logger import Logger

logger = Logger(module="tests/test_id3.py")


def test_build_id3_with_no_config():
    df = pd.read_csv("dataset/golf.txt")
    model = cb.fit(df, silent=True)
    assert model["config"]["algorithm"] == "ID3"
    logger.info("✅ standard id3 test done")


def test_build_id3_with_internal_validation_df():
    df = pd.read_csv("dataset/golf.txt")
    validation_df = pd.read_csv("dataset/golf.txt")

    model = cb.fit(df, validation_df=validation_df, silent=True)

    assert model["config"]["algorithm"] == "ID3"

    validation_eval_results = model["evaluation"]["validation"]

    assert validation_eval_results.get("Accuracy", 0) > 99
    assert validation_eval_results.get("Precision", 0) > 99
    assert validation_eval_results.get("Recall", 0) > 99
    assert validation_eval_results.get("F1", 0) > 99
    assert validation_eval_results.get("Instances", 0) == validation_df.shape[0]
    assert "Confusion matrix" in validation_eval_results.keys()
    assert "Labels" in validation_eval_results.keys()

    # decision_rules = model["trees"][0].__dict__["__name__"]+".py"
    decision_rules = model["trees"][0].__dict__["__spec__"].origin

    fi_df = cb.feature_importance(decision_rules, silent=True)
    assert fi_df.shape[0] == 4

    logger.info("✅ id3 test with internal validation data frame done")


def test_build_id3_with_external_validation_set():
    df = pd.read_csv("dataset/golf.txt")
    model = cb.fit(df, silent=True)

    assert model["config"]["algorithm"] == "ID3"

    validation_df = pd.read_csv("dataset/golf.txt")
    results = cb.evaluate(model, validation_df, silent=True)

    assert results.get("Accuracy", 0) > 99
    assert results.get("Precision", 0) > 99
    assert results.get("Recall", 0) > 99
    assert results.get("F1", 0) > 99
    assert results.get("Instances", 0) == validation_df.shape[0]
    assert "Confusion matrix" in results.keys()
    assert "Labels" in results.keys()

    logger.info("✅ id3 test with external validation data frame done")


def test_model_restoration():
    df = pd.read_csv("dataset/golf.txt")
    model = cb.fit(df, silent=True)
    assert model["config"]["algorithm"] == "ID3"

    cb.save_model(model)

    restored_model = cb.load_model("model.pkl")

    assert restored_model["config"]["algorithm"] == "ID3"

    instance = ["Sunny", "Hot", "High", "Weak"]

    prediction = cb.predict(restored_model, instance)
    assert prediction == "No"

    logger.info("✅ id3 model restoration test done")


def test_build_id3_for_nominal_and_numeric_features_nominal_target():
    df = pd.read_csv("dataset/golf2.txt")
    model = cb.fit(df, silent=True)

    assert model["config"]["algorithm"] == "ID3"

    instance = ["Sunny", 85, 85, "Weak"]
    prediction = cb.predict(model, instance)
    assert prediction == "No"
    logger.info("✅ build id3 for nominal and numeric features and nominal target test done")


def test_large_data_set():
    df = pd.read_csv("dataset/car.data")
    model = cb.fit(df, silent=True)

    assert model["config"]["algorithm"] == "ID3"

    instance = ["vhigh", "vhigh", 2, "2", "small", "low"]
    prediction = cb.predict(model, instance)
    assert prediction == "unacc"

    instance = ["high", "high", "4", "more", "big", "high"]
    prediction = cb.predict(model, instance)
    assert prediction == "acc"


def test_iris_dataset():
    df = pd.read_csv(
        "dataset/iris.data",
        names=["Sepal length", "Sepal width", "Petal length", "Petal width", "Decision"],
    )
    model = cb.fit(df, silent=True)
    assert model["config"]["algorithm"] == "ID3"
