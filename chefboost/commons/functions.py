import pathlib
import os
import sys
from os import path
from types import ModuleType
import multiprocessing
from typing import Optional, Union
import numpy as np
import pandas as pd
from chefboost import Chefboost as cb
from chefboost.commons.logger import Logger
from chefboost.commons.module import load_module

# pylint: disable=no-else-return, broad-except

logger = Logger(module="chefboost/commons/functions.py")


def bulk_prediction(df: pd.DataFrame, model: dict) -> None:
    """
    Perform a bulk prediction on given dataframe
    Args:
        df (pd.DataFrame): input data frame
        model (dict): built model
    Returns:
        None
    """
    predictions = []
    for _, instance in df.iterrows():
        features = instance.values[0:-1]
        prediction = cb.predict(model, features)
        predictions.append(prediction)

    df["Prediction"] = predictions


def restoreTree(module_name: str) -> ModuleType:
    """
    Restores a built tree
    """
    return load_module(module_name)


def softmax(w: list) -> np.ndarray:
    """
    Softmax function
    Args:
        w (list): probabilities
    Returns:
        result (numpy.ndarray): softmax of inputs
    """
    e = np.exp(np.array(w, dtype=np.float32))
    dist = e / np.sum(e)
    return dist


def sign(x: Union[int, float]) -> int:
    """
    Sign function
    Args:
        x (int or float): input
    Returns
        result (int) 1 for positive inputs, -1 for negative
            inputs, 0 for neutral input
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def formatRule(root: int) -> str:
    """
    Format a rule in the output file (tree)
    Args:
        root (int): degree of current rule
    Returns:
        formatted rule (str)
    """
    resp = ""

    for _ in range(0, root):
        resp = resp + "   "

    return resp


def storeRule(file: str, content: str) -> None:
    """
    Store a custom rule
    Args:
        file (str): target file
        content (str): content to store
    Returns:
        None
    """
    with open(file, "a+", encoding="UTF-8") as f:
        f.writelines(content)
        f.writelines("\n")


def createFile(file: str, content: str) -> None:
    """
    Create a file with given content
    Args:
        file (str): target file
        content (str): content to store
    Returns
        None
    """
    with open(file, "w", encoding="UTF-8") as f:
        f.write(content)


def initializeFolders() -> None:
    """
    Initialize required folders
    """
    sys.path.append("..")
    pathlib.Path("outputs").mkdir(parents=True, exist_ok=True)
    pathlib.Path("outputs/data").mkdir(parents=True, exist_ok=True)
    pathlib.Path("outputs/rules").mkdir(parents=True, exist_ok=True)

    # -----------------------------------

    # clear existing rules in outputs/

    outputs_path = os.getcwd() + os.path.sep + "outputs" + os.path.sep

    try:
        if path.exists(outputs_path + "data"):
            for file in os.listdir(outputs_path + "data"):
                os.remove(outputs_path + "data" + os.path.sep + file)

        if path.exists(outputs_path + "rules"):
            for file in os.listdir(outputs_path + "rules"):
                if (
                    ".py" in file
                    or ".json" in file
                    or ".txt" in file
                    or ".pkl" in file
                    or ".csv" in file
                ):
                    os.remove(outputs_path + "rules" + os.path.sep + file)
    except Exception as err:
        logger.warn(str(err))

    # ------------------------------------


def initializeParams(config: Optional[dict] = None) -> dict:
    """
    Arrange a chefboost configuration
    Args:
        config (dict): initial configuration
    Returns:
        config (dict): final configuration
    """
    if config is None:
        config = {}

    # set these default values if they are not mentioned in config
    if config.get("algorithm") is None:
        config["algorithm"] = "ID3"

    if config.get("enableRandomForest") is None:
        config["enableRandomForest"] = False

    if config.get("num_of_trees") is None:
        config["num_of_trees"] = 5

    if config.get("enableMultitasking") is None:
        config["enableMultitasking"] = False

    if config.get("enableGBM") is None:
        config["enableGBM"] = False

    if config.get("epochs") is None:
        config["epochs"] = 10

    if config.get("learning_rate") is None:
        config["learning_rate"] = 1

    if config.get("max_depth") is None:
        config["max_depth"] = 5

    if config.get("enableAdaboost") is None:
        config["enableAdaboost"] = False

    if config.get("num_of_weak_classifier") is None:
        config["num_of_weak_classifier"] = 4

    if config.get("enableParallelism") is None:
        config["enableParallelism"] = False

    if config.get("num_cores") is None:
        # allocate half of your total cores
        config["num_cores"] = int(multiprocessing.cpu_count() / 2)

    return config
