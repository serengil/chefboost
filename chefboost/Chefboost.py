import time
import pickle
import os
import json
from typing import Optional, Dict, Any, Union

import numpy as np
import pandas as pd

from chefboost.commons import functions, evaluate as cb_eval
from chefboost.training import Training
from chefboost.tuning import gbm, adaboost as adaboost_clf, randomforest
from chefboost.commons.logger import Logger

# pylint: disable=too-many-nested-blocks, no-else-return, inconsistent-return-statements

logger = Logger(module="chefboost/Chefboost.py")

# ------------------------


def fit(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    target_label: str = "Decision",
    validation_df: Optional[pd.DataFrame] = None,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Build (a) decision tree model(s)

    Args:
            df (pandas data frame): Training data frame.

            config (dictionary): training configuration. e.g.

                    config = {
                            'algorithm' (string): ID3, 'C4.5, CART, CHAID or Regression
                            'enableParallelism' (boolean): False

                            'enableGBM' (boolean): True,
                            'epochs' (int): 7,
                            'learning_rate' (int): 1,

                            'enableRandomForest' (boolean): True,
                            'num_of_trees' (int): 5,

                            'enableAdaboost' (boolean): True,
                            'num_of_weak_classifier' (int): 4
                    }

            target_label (str): target label for supervised learning.
                Default is Decision at the end of dataframe.

            validation_df (pandas data frame): validation data frame
                if nothing is passed to validation data frame, then the function validates
                built trees for training data frame

            silent (bool): set this to True if you do not want to see
                any informative logs

    Returns:
            chefboost model
    """

    # ------------------------

    process_id = os.getpid()

    # ------------------------
    # rename target column name
    if target_label != "Decision":
        # TODO: what if another column name is Decision?
        df = df.rename(columns={target_label: "Decision"})

    # if target is not the last column
    if df.columns[-1] != "Decision":
        if "Decision" in df.columns:
            new_column_order = df.columns.drop("Decision").tolist() + ["Decision"]
            logger.debug(new_column_order)
            df = df[new_column_order]
        else:
            raise ValueError("Please set the target_label")

    # ------------------------

    base_df = df.copy()

    # ------------------------

    target_label = df.columns[len(df.columns) - 1]

    # ------------------------
    # handle NaN values

    nan_values = []

    for column in df.columns:
        if df[column].dtypes != "object":
            min_value = df[column].min()
            idx = df[df[column].isna()].index

            nan_value = []
            nan_value.append(column)

            if idx.shape[0] > 0:
                df.loc[idx, column] = min_value - 1
                nan_value.append(min_value - 1)
                logger.debug("NaN values are replaced to {min_value - 1} in column {column}")
            else:
                nan_value.append(None)

            nan_values.append(nan_value)

    # ------------------------

    # initialize params and folders
    config = functions.initializeParams(config)
    functions.initializeFolders()

    # ------------------------

    algorithm = config["algorithm"]

    valid_algorithms = ["ID3", "C4.5", "CART", "CHAID", "Regression"]

    if algorithm not in valid_algorithms:
        raise ValueError(
            "Invalid algorithm passed. You passed ",
            algorithm,
            " but valid algorithms are ",
            valid_algorithms,
        )

    # ------------------------

    enableRandomForest = config["enableRandomForest"]
    enableGBM = config["enableGBM"]
    enableAdaboost = config["enableAdaboost"]
    enableParallelism = config["enableParallelism"]

    # ------------------------

    if enableParallelism == True:
        num_cores = config["num_cores"]
        if silent is False:
            logger.info(f"[INFO]: {num_cores} CPU cores will be allocated in parallel running")

        from multiprocessing import set_start_method, freeze_support

        set_start_method("spawn", force=True)
        freeze_support()
    # ------------------------
    num_of_columns = df.shape[1]

    if algorithm == "Regression":
        if df["Decision"].dtypes == "object":
            raise ValueError(
                "Regression trees cannot be applied for nominal target values!"
                "You can either change the algorithm or data set."
            )

    if (
        df["Decision"].dtypes != "object"
    ):  # this must be regression tree even if it is not mentioned in algorithm
        if algorithm != "Regression":
            logger.warn(
                f"You set the algorithm to {algorithm} but the Decision column of your"
                " data set has non-object type."
                "That's why, the algorithm is set to Regression to handle the data set."
            )

        algorithm = "Regression"
        config["algorithm"] = "Regression"

    if enableGBM == True:
        if silent is False:
            logger.info("Gradient Boosting Machines...")
        algorithm = "Regression"
        config["algorithm"] = "Regression"

    if enableAdaboost == True:
        # enableParallelism = False
        for j in range(0, num_of_columns):
            column_name = df.columns[j]
            if df[column_name].dtypes == "object":
                raise ValueError(
                    "Adaboost must be run on numeric data set for both features and target"
                )

    # -------------------------

    if silent is False:
        logger.info(f"{algorithm} tree is going to be built...")

    # initialize a dictionary. this is going to be used to check features numeric or nominal.
    # numeric features should be transformed to nominal values based on scales.
    dataset_features = {}

    header = "def findDecision(obj): #"

    num_of_columns = df.shape[1] - 1
    for i in range(0, num_of_columns):
        column_name = df.columns[i]
        dataset_features[column_name] = df[column_name].dtypes
        header += f"obj[{str(i)}]: {column_name}"

        if i != num_of_columns - 1:
            header = header + ", "

    header = header + "\n"

    # ------------------------

    begin = time.time()

    trees = []
    alphas = []

    if enableAdaboost == True:
        trees, alphas = adaboost_clf.apply(
            df,
            config,
            header,
            dataset_features,
            validation_df=validation_df,
            process_id=process_id,
            silent=silent,
        )

    elif enableGBM == True:
        if df["Decision"].dtypes == "object":  # transform classification problem to regression
            trees, alphas = gbm.classifier(
                df,
                config,
                header,
                dataset_features,
                validation_df=validation_df,
                process_id=process_id,
                silent=silent,
            )
            # classification = True

        else:  # regression
            trees = gbm.regressor(
                df,
                config,
                header,
                dataset_features,
                validation_df=validation_df,
                process_id=process_id,
                silent=silent,
            )
            # classification = False

    elif enableRandomForest == True:
        trees = randomforest.apply(
            df,
            config,
            header,
            dataset_features,
            validation_df=validation_df,
            process_id=process_id,
            silent=silent,
        )
    else:  # regular decision tree building
        root = 1
        file = "outputs/rules/rules.py"
        functions.createFile(file, header)

        if enableParallelism == True:
            json_file = "outputs/rules/rules.json"
            functions.createFile(json_file, "[\n")

        trees = Training.buildDecisionTree(
            df,
            root=root,
            file=file,
            config=config,
            dataset_features=dataset_features,
            parent_level=0,
            leaf_id=0,
            parents="root",
            validation_df=validation_df,
            main_process_id=process_id,
        )

    if silent is False:
        logger.info("-------------------------")
        logger.info(f"finished in {time.time() - begin} seconds")

    obj = {"trees": trees, "alphas": alphas, "config": config, "nan_values": nan_values}

    # -----------------------------------------

    # train set accuracy
    df = base_df.copy()
    trainset_evaluation = evaluate(obj, df, task="train", silent=silent)
    obj["evaluation"] = {"train": trainset_evaluation}

    # validation set accuracy
    if isinstance(validation_df, pd.DataFrame):
        validationset_evaluation = evaluate(obj, validation_df, task="validation", silent=silent)
        obj["evaluation"]["validation"] = validationset_evaluation

    return obj

    # -----------------------------------------


def predict(model: dict, param: list) -> Union[str, int, float]:
    """
    Predict the target label of given features from a pre-trained model
    Args:
        model (built chefboost model): pre-trained model which is the output
            of fit function
        param (list): pass input features as python list
            e.g. chef.predict(model, param = ['Sunny', 'Hot', 'High', 'Weak'])
    Returns:
            prediction
    """

    trees = model["trees"]
    config = model["config"]

    alphas = []
    if "alphas" in model:
        alphas = model["alphas"]

    nan_values = []
    if "nan_values" in model:
        nan_values = model["nan_values"]

    # -----------------------
    # handle missing values

    column_index = 0
    for column in nan_values:
        column_name = column[0]
        missing_value = column[1]

        if pd.isna(missing_value) != True:
            logger.debug(
                f"missing values will be replaced with {missing_value} in {column_name} column"
            )

            if pd.isna(param[column_index]):
                param[column_index] = missing_value

        column_index = column_index + 1

    logger.debug(f"instance: {param}")
    # -----------------------

    enableGBM = config["enableGBM"]
    adaboost = config["enableAdaboost"]
    enableRandomForest = config["enableRandomForest"]

    # -----------------------

    classification = False
    prediction = 0
    prediction_classes = []

    # -----------------------

    if enableGBM == True:
        if len(trees) == config["epochs"]:
            classification = False
        else:
            classification = True
            prediction_classes = [0 for i in alphas]

    # -----------------------

    if len(trees) > 1:  # bagging or boosting
        index = 0
        for tree in trees:
            if adaboost != True:
                custom_prediction = tree.findDecision(param)

                if custom_prediction != None:
                    if not isinstance(custom_prediction, str):  # regression
                        if enableGBM == True and classification == True:
                            prediction_classes[index % len(alphas)] += custom_prediction
                        else:
                            prediction += custom_prediction
                    else:
                        classification = True
                        prediction_classes.append(custom_prediction)
            else:  # adaboost
                prediction += alphas[index] * tree.findDecision(param)
            index = index + 1

        if enableRandomForest == True:
            # notice that gbm requires cumilative sum but random forest requires mean of each tree
            prediction = prediction / len(trees)

        if adaboost == True:
            prediction = functions.sign(prediction)
    else:  # regular decision tree
        tree = trees[0]
        prediction = tree.findDecision(param)

    if classification == False:
        return prediction
    else:
        if enableGBM == True and classification == True:
            return alphas[np.argmax(prediction_classes)]
        else:  # classification
            # e.g. random forest
            # get predictions made by different trees
            predictions = np.array(prediction_classes)

            # find the most frequent prediction
            (values, counts) = np.unique(predictions, return_counts=True)
            idx = np.argmax(counts)
            prediction = values[idx]

            return prediction


def save_model(base_model: dict, file_name: str = "model.pkl") -> None:
    """
    Save pre-trained model on file system
    Args:
            base_model (dict): pre-trained model which is the output
                of the fit function
            file_name (string): target file name as exact path.
    """

    model = base_model.copy()

    # modules cannot be saved. Save its reference instead.
    module_names = []
    for tree in model["trees"]:
        module_names.append(tree.__name__)

    model["trees"] = module_names

    with open(f"outputs/rules/{file_name}", "wb") as f:
        pickle.dump(model, f)


def load_model(file_name: str = "model.pkl") -> dict:
    """
    Load the save pre-trained model from file system
    Args:
            file_name (str): exact path of the target saved model
    Returns:
            built model (dict)
    """

    with open("outputs/rules/" + file_name, "rb") as f:
        model = pickle.load(f)

    # restore modules from its references
    modules = []
    for model_name in model["trees"]:
        module = functions.restoreTree(model_name)
        modules.append(module)

    model["trees"] = modules

    return model


def restoreTree(module_name) -> Any:
    """
    Load built model from set of decision rules
    Args:
        module_name (str): e.g. outputs/rules/rules to restore outputs/rules/rules.py
    Returns:
            built model (dict)
    """

    return functions.restoreTree(module_name)


def feature_importance(rules: Union[str, list], silent: bool = False) -> pd.DataFrame:
    """
    Show the feature importance values of a built model
    Args:
        rules (str or list): e.g. decision_rules = "outputs/rules/rules.py"
            or this could be retrieved from built model as shown below.

            ```python
            decision_rules = []
            for tree in model["trees"]:
               rule = .__dict__["__spec__"].origin
               decision_rules.append(rule)
            ```
        silent (bool): set this to True if you do want to see
            any informative logs.
    Returns:
            feature importance (pd.DataFrame)
    """

    if not isinstance(rules, list):
        rules = [rules]

    if silent is False:
        logger.info(f"rules: {rules}")

    # -----------------------------

    dfs = []

    for rule in rules:
        if silent is False:
            logger.info(f"Decision rule: {rule}")

        with open(rule, "r", encoding="UTF-8") as file:
            lines = file.readlines()

        pivot = {}
        rules = []

        # initialize feature importances
        line_idx = 0
        for line in lines:
            if line_idx == 0:
                feature_explainer_list = line.split("#")[1].split(", ")
                for feature_explainer in feature_explainer_list:
                    feature = feature_explainer.split(": ")[1].replace("\n", "")
                    pivot[feature] = 0
            else:
                if "# " in line:
                    rule = line.strip().split("# ")[1]
                    rules.append(json.loads(rule))

            line_idx = line_idx + 1

        feature_names = list(pivot.keys())

        for feature in feature_names:
            for rule in rules:
                if rule["feature"] == feature:
                    score = rule["metric_value"] * rule["instances"]
                    current_depth = rule["depth"]

                    child_scores = 0
                    # find child node importances
                    for child_rule in rules:
                        if child_rule["depth"] == current_depth + 1:
                            child_score = child_rule["metric_value"] * child_rule["instances"]

                            child_scores = child_scores + child_score

                    score = score - child_scores

                    pivot[feature] = pivot[feature] + score

        # normalize feature importance

        total_score = 0
        for feature, score in pivot.items():
            total_score = total_score + score

        for feature, score in pivot.items():
            pivot[feature] = round(pivot[feature] / total_score, 4)

        instances = []
        for feature, score in pivot.items():
            instance = []
            instance.append(feature)
            instance.append(score)
            instances.append(instance)

        df = pd.DataFrame(instances, columns=["feature", "final_importance"])
        df = df.sort_values(by=["final_importance"], ascending=False)

        if len(rules) == 1:
            return df
        else:
            dfs.append(df)

    if len(rules) > 1:
        hf = pd.DataFrame(feature_names, columns=["feature"])
        hf["importance"] = 0

        for df in dfs:
            hf = hf.merge(df, on=["feature"], how="left")
            hf["importance"] = hf["importance"] + hf["final_importance"]
            hf = hf.drop(columns=["final_importance"])

        # ------------------------
        # normalize
        hf["importance"] = hf["importance"] / hf["importance"].sum()
        hf = hf.sort_values(by=["importance"], ascending=False)

        return hf


def evaluate(
    model: dict,
    df: pd.DataFrame,
    target_label: str = "Decision",
    task: str = "test",
    silent: bool = False,
) -> dict:
    """
    Evaluate the performance of a built model on a data set
    Args:
        model (dict): built model which is the output of fit function
        df (pandas data frame): data frame you would like to evaluate
        target_label (str): target label
        task (string): set this to train, validation or test
        silent (bool): set this to True if you do not want to see
            any informative logs
    Returns:
        evaluation results (dict)
    """

    # --------------------------

    if target_label != "Decision":
        df = df.rename(columns={target_label: "Decision"})

    # if target is not the last column
    if df.columns[-1] != "Decision":
        new_column_order = df.columns.drop("Decision").tolist() + ["Decision"]
        logger.debug(new_column_order)
        df = df[new_column_order]

    # --------------------------

    functions.bulk_prediction(df, model)

    enableAdaboost = model["config"]["enableAdaboost"]

    if enableAdaboost == True:
        df["Decision"] = df["Decision"].astype(str)
        df["Prediction"] = df["Prediction"].astype(str)

    return cb_eval.evaluate(df, task=task, silent=silent)
