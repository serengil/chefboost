import math
from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from chefboost.commons import functions
from chefboost.training import Training
from chefboost.commons.logger import Logger
from chefboost.commons.module import load_module

# pylint: disable=unused-argument

logger = Logger(module="chefboost/tuning/adaboost.py")


def findPrediction(row: pd.Series) -> int:
    """
    Make prediction for an instance with a built adaboost model
    Args:
        row (pd.Series): row of a pandas dataframe
    Returns
        prediction (int)
    """
    epoch = row["Epoch"]
    row = row.drop(labels=["Epoch"])
    columns = row.shape[0]

    params = []
    for j in range(0, columns - 1):
        params.append(row[j])

    module_name = f"outputs/rules/rules_{int(epoch)}"
    myrules = load_module(module_name)

    prediction = functions.sign(myrules.findDecision(params))

    return prediction


def apply(
    df: pd.DataFrame,
    config: dict,
    header: str,
    dataset_features: dict,
    validation_df: Optional[pd.DataFrame] = None,
    process_id: Optional[int] = None,
    silent: bool = False,
) -> tuple:
    """
    Train procedure of adaboost algorithm
    Args:
        df (pd.DataFrame): train set
        config (dict): configuration sent to fit function
        header (str): output module's header line
        dataset_features (dict): dataframe's columns with datatypes
        validation_df (pd.DataFrame): validation set
        process_id (int): process id of parent trx
        silent (bool): set this to True to make it silent
    Returns:
        result (tuple): models and alphas
    """
    models = []
    alphas = []

    initializeAlphaFile()

    num_of_weak_classifier = config["num_of_weak_classifier"]

    # ------------------------

    rows = df.shape[0]
    final_predictions = pd.DataFrame(np.zeros([rows, 1]), columns=["prediction"])

    worksheet = df.copy()
    worksheet["Weight"] = 1 / rows  # uniform distribution initially

    final_predictions = pd.DataFrame(np.zeros((df.shape[0], 2)), columns=["Prediction", "Actual"])
    final_predictions["Actual"] = df["Decision"]

    best_epoch_idx = 0
    best_epoch_value = 1000000

    pbar = tqdm(range(0, num_of_weak_classifier), desc="Adaboosting", disable=silent)
    for i in pbar:
        worksheet["Decision"] = worksheet["Weight"] * worksheet["Decision"]

        root = 1
        file = "outputs/rules/rules_" + str(i) + ".py"

        functions.createFile(file, header)

        logger.debug(worksheet)
        Training.buildDecisionTree(
            worksheet.drop(columns=["Weight"]),
            root,
            file,
            config,
            dataset_features,
            parent_level=0,
            leaf_id=0,
            parents="root",
            main_process_id=process_id,
        )

        # ---------------------------------------

        module_name = "outputs/rules/rules_" + str(i)
        myrules = load_module(module_name)
        models.append(myrules)

        # ---------------------------------------

        df["Epoch"] = i
        worksheet["Prediction"] = df.apply(findPrediction, axis=1)
        df = df.drop(columns=["Epoch"])

        # ---------------------------------------
        worksheet["Actual"] = df["Decision"]
        worksheet["Loss"] = abs(worksheet["Actual"] - worksheet["Prediction"]) / 2
        worksheet["Weight_Times_Loss"] = worksheet["Loss"] * worksheet["Weight"]

        epsilon = worksheet["Weight_Times_Loss"].sum()
        alpha = (
            math.log((1 - epsilon) / epsilon) / 2
        )  # use alpha to update weights in the next round
        alphas.append(alpha)

        # -----------------------------

        # store alpha
        addEpochAlpha(i, alpha)

        # -----------------------------

        worksheet["Alpha"] = alpha
        worksheet["New_Weights"] = worksheet["Weight"] * (
            -alpha * worksheet["Actual"] * worksheet["Prediction"]
        ).apply(math.exp)

        # normalize
        worksheet["New_Weights"] = worksheet["New_Weights"] / worksheet["New_Weights"].sum()
        worksheet["Weight"] = worksheet["New_Weights"]
        worksheet["Decision"] = df["Decision"]

        final_predictions["Prediction"] = (
            final_predictions["Prediction"] + worksheet["Alpha"] * worksheet["Prediction"]
        )
        logger.debug(final_predictions)
        worksheet = worksheet.drop(
            columns=["New_Weights", "Prediction", "Actual", "Loss", "Weight_Times_Loss", "Alpha"]
        )

        mae = (
            np.abs(
                final_predictions["Prediction"].apply(functions.sign) - final_predictions["Actual"]
            )
            / 2
        ).sum() / final_predictions.shape[0]
        logger.debug(mae)

        if mae < best_epoch_value:
            best_epoch_value = mae * 1
            best_epoch_idx = i * 1

        pbar.set_description(f"Epoch {i + 1}. Loss: {mae}. Process: ")

    # ------------------------------
    if silent is False:
        logger.info(f"The best epoch is {best_epoch_idx} with the {best_epoch_value} MAE score")

    models = models[0 : best_epoch_idx + 1]
    alphas = alphas[0 : best_epoch_idx + 1]

    # ------------------------------

    return models, alphas


def initializeAlphaFile() -> None:
    """
    Initialize alpha file
    """
    file = "outputs/rules/alphas.py"
    header = "def findAlpha(epoch):\n"
    functions.createFile(file, header)


def addEpochAlpha(epoch: int, alpha: int) -> None:
    """
    Add epoch's result into alpha file
    """
    file = "outputs/rules/alphas.py"
    content = "   if epoch == " + str(epoch) + ":\n"
    content += "      return " + str(alpha)
    functions.storeRule(file, content)
