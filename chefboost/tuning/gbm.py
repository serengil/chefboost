import gc
from typing import Optional, Union

import pandas as pd
import numpy as np
from tqdm import tqdm

from chefboost.commons import functions
from chefboost.training import Training
from chefboost.commons.logger import Logger
from chefboost.commons.module import load_module

# pylint: disable=unused-argument

logger = Logger(module="chefboost/tuning/gbm.py")


def findPrediction(row: pd.Series) -> Union[str, int, float]:
    """
    Make a prediction for a row of data frame on built gbm model
    Args:
        row (pd.Series): a row of a pandas data frame
    Returns:
        result (str or float): str for classifier, int or float
            for regressor
    """
    epoch = row["Epoch"]
    row = row.drop(labels=["Epoch"])
    columns = row.shape[0]

    params = []
    for j in range(0, columns - 1):
        params.append(row[j])

    module_name = f"outputs/rules/rules{epoch - 1}"
    myrules = load_module(module_name)

    # prediction = int(myrules.findDecision(params))
    prediction = myrules.findDecision(params)

    return prediction


def regressor(
    df: pd.DataFrame,
    config: dict,
    header: str,
    dataset_features: dict,
    validation_df: Optional[pd.DataFrame] = None,
    process_id: Optional[int] = None,
    silent: bool = False,
) -> list:
    """
    Train procedure of adaboost gbm regressor
    Args:
        df (pd.DataFrame): train set
        config (dict): configuration sent to fit function
        header (str): output module's header line
        dataset_features (dict): dataframe's columns with datatypes
        validation_df (pd.DataFrame): validation set
        process_id (int): process id of parent trx
        silent (bool): set this to True to make it silent
    Returns:
        result (list): list of built models
    """
    models = []

    # we will update decisions in every epoch, this will be used to restore
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]

    boosted_from = 0
    boosted_to = 0

    base_df = df.copy()

    # gbm will manipulate actuals. store its raw version.
    target_values = base_df["Decision"].values
    num_of_instances = target_values.shape[0]

    root = 1
    file = "outputs/rules/rules0.py"
    json_file = "outputs/rules/rules0.json"
    functions.createFile(file, header)
    functions.createFile(json_file, "[\n")

    Training.buildDecisionTree(
        df, root, file, config, dataset_features, parent_level=0, leaf_id=0, parents="root"
    )  # generate rules0

    # functions.storeRule(json_file," {}]")

    df = base_df.copy()

    base_df["Boosted_Prediction"] = 0

    # ------------------------------

    best_epoch_idx = 0
    best_epoch_loss = 1000000

    pbar = tqdm(range(1, epochs + 1), desc="Boosting", disable=silent)
    for index in pbar:
        logger.debug(f"epoch {index} - ")
        loss = 0

        # run data(i-1) and rules(i-1), save data1

        # dynamic import
        module_name = f"outputs/rules/rules{index - 1}"
        myrules = load_module(module_name)  # rules0

        models.append(myrules)

        new_data_set = f"outputs/data/data{index}.csv"
        with open(new_data_set, "w", encoding="UTF-8"):
            pass

        # ----------------------------------------

        df["Epoch"] = index
        df["Prediction"] = df.apply(findPrediction, axis=1)

        base_df["Boosted_Prediction"] += df["Prediction"]

        loss = (base_df["Boosted_Prediction"] - base_df["Decision"]).pow(2).sum()
        current_loss = loss / num_of_instances  # mse

        if index == 1:
            boosted_from = current_loss * 1
        elif index == epochs:
            boosted_to = current_loss * 1
            logger.debug(f"Boosted to {boosted_to}")

        if current_loss < best_epoch_loss:
            best_epoch_loss = current_loss * 1
            best_epoch_idx = index * 1

        df["Decision"] = int(learning_rate) * (df["Decision"] - df["Prediction"])
        df = df.drop(columns=["Epoch", "Prediction"])

        # ---------------------------------

        df.to_csv(new_data_set, index=False)
        # data(i) created

        # ---------------------------------

        file = "outputs/rules/rules" + str(index) + ".py"
        json_file = "outputs/rules/rules" + str(index) + ".json"

        functions.createFile(file, header)
        functions.createFile(json_file, "[\n")

        current_df = df.copy()
        Training.buildDecisionTree(
            df,
            root,
            file,
            config,
            dataset_features,
            parent_level=0,
            leaf_id=0,
            parents="root",
            main_process_id=process_id,
        )

        # functions.storeRule(json_file," {}]")

        df = (
            current_df.copy()
        )  # numeric features require this restoration to apply findDecision function

        # rules(i) created

        loss = loss / num_of_instances
        logger.debug(f"epoch {index} - loss: {loss}")
        logger.debug(f"loss: {loss}")
        pbar.set_description(f"Epoch {index}. Loss: {loss}. Process: ")

        gc.collect()

    # ---------------------------------

    if silent is False:
        logger.info(f"The best epoch is {best_epoch_idx} with {best_epoch_loss} loss value")
    models = models[0:best_epoch_idx]
    config["epochs"] = best_epoch_idx

    if silent is False:
        logger.info(
            f"MSE of {num_of_instances} instances are boosted from {boosted_from}"
            f"to {best_epoch_loss} in {epochs} epochs"
        )

    return models


def classifier(
    df: pd.DataFrame,
    config: dict,
    header: str,
    dataset_features: dict,
    validation_df: Optional[pd.DataFrame] = None,
    process_id: Optional[int] = None,
    silent: bool = False,
) -> tuple:
    """
    Train procedure of adaboost gbm classifier
    Args:
        df (pd.DataFrame): train set
        config (dict): configuration sent to fit function
        header (str): output module's header line
        dataset_features (dict): dataframe's columns with datatypes
        validation_df (pd.DataFrame): validation set
        process_id (int): process id of parent trx
        silent (bool): set this to True to make it silent
    Returns:
        result (tuple): list of built models, unique classes
            in target column
    """
    models = []

    if silent is False:
        logger.info("gradient boosting for classification")

    epochs = config["epochs"]
    enableParallelism = config["enableParallelism"]

    temp_df = df.copy()
    worksheet = df.copy()

    classes = df["Decision"].unique()

    boosted_predictions = np.zeros([df.shape[0], len(classes)])

    pbar = tqdm(range(0, epochs), desc="Boosting", disable=silent)

    # store actual set, we will use this to calculate loss
    actual_set = pd.DataFrame(np.zeros([df.shape[0], len(classes)]), columns=classes)
    for current_class in classes:
        actual_set[current_class] = np.where(df["Decision"] == current_class, 1, 0)
    actual_set = actual_set.values  # transform it to numpy array

    best_accuracy_idx = 0
    best_accuracy_value = 0
    accuracies = []

    # for epoch in range(0, epochs):
    for epoch in pbar:
        for i, current_class in enumerate(classes):

            if epoch == 0:
                temp_df["Decision"] = np.where(df["Decision"] == current_class, 1, 0)
                worksheet["Y_" + str(i)] = temp_df["Decision"]
            else:
                temp_df["Decision"] = worksheet["Y-P_" + str(i)]

            predictions = []

            # change data type for decision column
            temp_df[["Decision"]].astype("int64")

            root = 1
            file_base = "outputs/rules/rules-for-" + current_class + "-round-" + str(epoch)

            file = file_base + ".py"
            functions.createFile(file, header)

            if enableParallelism == True:
                json_file = file_base + ".json"
                functions.createFile(json_file, "[\n")

            Training.buildDecisionTree(
                temp_df,
                root,
                file,
                config,
                dataset_features,
                parent_level=0,
                leaf_id=0,
                parents="root",
                main_process_id=process_id,
            )

            # decision rules created
            # ----------------------------

            # dynamic import
            module_name = "outputs/rules/rules-for-" + current_class + "-round-" + str(epoch)
            myrules = load_module(module_name)  # rules0

            models.append(myrules)

            num_of_columns = df.shape[1]

            for row, instance in df.iterrows():
                features = []
                for j in range(0, num_of_columns - 1):  # iterate on features
                    features.append(instance[j])

                actual = temp_df.loc[row]["Decision"]
                prediction = myrules.findDecision(features)

                predictions.append(prediction)

            # ----------------------------
            if epoch == 0:
                worksheet["F_" + str(i)] = 0
            else:
                worksheet["F_" + str(i)] = pd.Series(predictions).values

            boosted_predictions[:, i] = boosted_predictions[:, i] + worksheet[
                "F_" + str(i)
            ].values.astype(np.float32)

            logger.debug(boosted_predictions[0:5, :])

            worksheet["P_" + str(i)] = 0

            # ----------------------------
            temp_df = df.copy()  # restoration

        for row, instance in worksheet.iterrows():
            f_scores = []
            for i in range(0, len(classes)):
                f_scores.append(instance["F_" + str(i)])

            probabilities = functions.softmax(f_scores)

            for j, current_prob in enumerate(probabilities):
                instance["P_" + str(j)] = current_prob

            worksheet.loc[row] = instance

        for i in range(0, len(classes)):
            worksheet["Y-P_" + str(i)] = worksheet["Y_" + str(i)] - worksheet["P_" + str(i)]

        prediction_set = np.zeros([df.shape[0], len(classes)])
        for i in range(0, boosted_predictions.shape[0]):
            predicted_index = np.argmax(boosted_predictions[i])
            prediction_set[i][predicted_index] = 1

        # ----------------------------
        # find loss for this epoch: prediction_set vs actual_set
        classified = 0
        for i in range(0, actual_set.shape[0]):
            actual = np.argmax(actual_set[i])
            prediction = np.argmax(prediction_set[i])
            logger.debug(f"actual: {actual} - prediction: {prediction}")

            if actual == prediction:
                classified = classified + 1

        accuracy = 100 * classified / actual_set.shape[0]
        accuracies.append(accuracy)

        if accuracy > best_accuracy_value:
            best_accuracy_value = accuracy * 1
            best_accuracy_idx = epoch * 1

        # ----------------------------

        logger.debug(worksheet.head())
        logger.debug("round {epoch+1}")
        pbar.set_description(f"Epoch {epoch + 1}. Accuracy: {accuracy}. Process: ")

        gc.collect()

    # --------------------------------

    if silent is False:
        logger.info(
            f"The best accuracy got in {best_accuracy_idx} epoch"
            f" with the score {best_accuracy_value}"
        )

    models = models[0 : best_accuracy_idx * len(classes) + len(classes)]

    return models, classes
