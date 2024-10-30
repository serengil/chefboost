# built-in imports
import math
import uuid
import json
import copy
import gc
from contextlib import closing
from typing import Optional

# 3rd party
import psutil
import numpy as np
import pandas as pd

# project's imports
from chefboost.training import Preprocess
from chefboost.commons import functions
from chefboost.commons.logger import Logger
from chefboost.commons.module import load_module
from chefboost.commons.daemon import CustomPool

# pylint: disable=too-many-function-args, unused-argument

logger = Logger(module="chefboost/training/Training.py")

global decision_rules  # pylint: disable=global-at-module-level


def calculateEntropy(df: pd.DataFrame, config: dict) -> float:
    """
    Calculate entropy for a given dataframe with respect
        to the algorithm in the configuration
    Args:
        df (pd.DataFrame): train set's data frame
        config (dict): configuration dictionary
    Returns:
        entropy (float)
    """
    algorithm = config["algorithm"]

    # --------------------------

    if algorithm == "Regression":
        return 0

    logger.debug(df)

    instances = df.shape[0]

    decisions = df["Decision"].value_counts().keys().tolist()

    entropy = 0
    for i, decision in enumerate(decisions):
        num_of_decisions = df["Decision"].value_counts().tolist()[i]
        logger.debug(f"{decision} -> {num_of_decisions}")
        class_probability = num_of_decisions / instances
        entropy = entropy - class_probability * math.log(class_probability, 2)

    return entropy


def findDecision(df: pd.DataFrame, config: dict) -> tuple:
    """
    Find the most dominant decision for a given dataframe
    Args:
        df (pd.DataFrame): (sub) train set's dataframe
        config (dict): configuration
    Returns
        result (tuple): the most dominant feature, data frame's
            number of instances, metric value (e.g. information gain
            for ID3), metric name (e.g. gain ratio for C4.5)

    Information about metrics: information gain for id3, gain ratio
    for c4.5, gini for cart,     chi square for chaid and std
    for regression
    """

    algorithm = config["algorithm"]

    resp_obj = findGains(df, config)
    gains = list(resp_obj["gains"].values())
    entropy = resp_obj["entropy"]

    if algorithm == "ID3":
        winner_index = gains.index(max(gains))
        metric_value = entropy
        metric_name = "Entropy"
    elif algorithm == "C4.5":
        winner_index = gains.index(max(gains))
        metric_value = entropy
        metric_name = "Entropy"
    elif algorithm == "CART":
        winner_index = gains.index(min(gains))
        metric_value = min(gains)
        metric_name = "Gini"
    elif algorithm == "CHAID":
        winner_index = gains.index(max(gains))
        metric_value = max(gains)
        metric_name = "ChiSquared"
    elif algorithm == "Regression":
        winner_index = gains.index(max(gains))
        metric_value = max(gains)
        metric_name = "Std"

    winner_name = df.columns[winner_index]

    return winner_name, df.shape[0], metric_value, metric_name


def findGains(df: pd.DataFrame, config: dict) -> dict:
    """
    Find entropy and gains of each feature in the data frame
    Args:
        df (pd.DataFrame): (sub) train set as data frame
        config (dict): training configuration
    Returns:
        result (dict): gains with respect to the algorithm
    """
    algorithm = config["algorithm"]
    decision_classes = df["Decision"].unique()

    # -----------------------------

    entropy = 0

    if algorithm in ["ID3", "C4.5"]:
        entropy = calculateEntropy(df, config)

    columns = df.shape[1]
    instances = df.shape[0]

    gains = []

    for i in range(0, columns - 1):
        column_name = df.columns[i]
        column_type = df[column_name].dtypes

        logger.debug(f"{column_name} -> {column_type}")

        if column_type != "object":
            df = Preprocess.processContinuousFeatures(algorithm, df, column_name, entropy, config)

        classes = df[column_name].value_counts()

        splitinfo = 0
        if algorithm in ["ID3", "C4.5"]:
            gain = entropy * 1
        else:
            gain = 0

        for j in range(0, len(classes)):
            current_class = classes.keys().tolist()[j]
            logger.debug(f"{column_name} -> {current_class}")

            subdataset = df[df[column_name] == current_class]
            logger.debug(subdataset)

            subset_instances = subdataset.shape[0]
            class_probability = subset_instances / instances

            if algorithm in ["ID3", "C4.5"]:
                subset_entropy = calculateEntropy(subdataset, config)
                gain = gain - class_probability * subset_entropy

            if algorithm == "C4.5":
                splitinfo = splitinfo - class_probability * math.log(class_probability, 2)

            elif algorithm == "CART":  # GINI index
                decision_list = subdataset["Decision"].value_counts().tolist()

                subgini = 1

                for current_decision in decision_list:
                    subgini = subgini - math.pow((current_decision / subset_instances), 2)

                gain = gain + (subset_instances / instances) * subgini

            elif algorithm == "CHAID":
                num_of_decisions = len(decision_classes)

                expected = subset_instances / num_of_decisions

                for d in decision_classes:
                    num_of_d = subdataset[subdataset["Decision"] == d].shape[0]

                    chi_square_of_d = math.sqrt(
                        ((num_of_d - expected) * (num_of_d - expected)) / expected
                    )

                    gain += chi_square_of_d

            elif algorithm == "Regression":
                subset_stdev = subdataset["Decision"].std(ddof=0)
                gain = gain + (subset_instances / instances) * subset_stdev

        # iterating over classes for loop end
        # -------------------------------

        if algorithm == "Regression":
            stdev = df["Decision"].std(ddof=0)
            gain = stdev - gain
        if algorithm == "C4.5":
            if splitinfo == 0:
                splitinfo = 100
                # this can be if data set consists of 2 rows and current column consists
                # of 1 class. still decision can be made (decisions for these 2 rows same).
                # set splitinfo to very large value to make gain ratio very small.
                # in this way, we won't find this column as the most dominant one.
            gain = gain / splitinfo

        # ----------------------------------

        gains.append(gain)

    # -------------------------------------------------

    resp_obj = {}
    resp_obj["gains"] = {}

    for idx, feature in enumerate(df.columns[0:-1]):  # Decision is always the last column
        logger.debug(f"{idx}, {feature}")
        resp_obj["gains"][feature] = gains[idx]

    resp_obj["entropy"] = entropy

    return resp_obj


def createBranchWrapper(func, args):
    return func(*args)


def createBranch(
    config: dict,
    current_class: str,
    subdataset: pd.DataFrame,
    numericColumn: bool,
    branch_index: int,
    winner_name: str,
    winner_index: int,
    root: int,
    parents: str,
    file: str,
    dataset_features: dict,
    num_of_instances: int,
    metric: float,
    tree_id: int = 0,
    main_process_id: Optional[int] = None,
) -> list:
    """
    Create a branch and its leafs in the tree
    Args:
        config (dict): training configuration
        current_class (str): condition for the current leaf
            e.g. '>40' or '<=40' if the current branch is age
        subdataset (pd.DataFrame): filtered dataframe for the
            current_class' condition
        numericColumn (bool): current branch is numeric or nominal
        branch_index (int): current index. will use if for the first
            one and elif for the rest
        winner_name (str): the most dominant feature's name
        winner_index (int): index of the most dominant feature's name
        root (int): depth of the current decision rule
        parents (str): parent's uuid. using root name for the top one.
        file (str): target file to store rules
        dataset_features (dict): feature names
        num_of_instances (int): number of rows
        metric (float): e.g. information gain result for ID3
        tree_id (int): current tree's id
        main_process_id (int): main tree's process id
    Returns
        custom rules (list)
    """
    custom_rules = []

    algorithm = config["algorithm"]
    enableAdaboost = config["enableAdaboost"]
    enableGBM = config["enableGBM"]
    max_depth = config["max_depth"]
    enableParallelism = config["enableParallelism"]

    charForResp = "'"
    if algorithm == "Regression":
        charForResp = ""

    # ---------------------------

    tmp_root = root * 1
    parents_raw = copy.copy(parents)

    # ---------------------------

    if numericColumn == True:
        compareTo = current_class  # current class might be <=x or >x in this case
    else:
        compareTo = " == '" + str(current_class) + "'"

    terminateBuilding = False

    # -----------------------------------------------
    # can decision be made?

    if enableGBM == True and root >= max_depth:  # max depth
        final_decision = subdataset["Decision"].mean()
        terminateBuilding = True
    elif enableAdaboost == True:
        # final_decision = subdataset['Decision'].value_counts().idxmax()
        final_decision = functions.sign(subdataset["Decision"].mean())  # get average
        terminateBuilding = True
        enableParallelism = False
    elif len(subdataset["Decision"].value_counts().tolist()) == 1:
        final_decision = (
            subdataset["Decision"].value_counts().keys().tolist()[0]
        )  # all items are equal in this case
        terminateBuilding = True
    elif subdataset.shape[1] == 1:  # if decision cannot be made even though all columns dropped
        final_decision = subdataset["Decision"].value_counts().idxmax()  # get the most frequent one
        terminateBuilding = True
    elif algorithm == "Regression" and (
        subdataset.shape[0] < 5 or root >= max_depth
    ):  # pruning condition
        final_decision = subdataset["Decision"].mean()  # get average
        terminateBuilding = True
    elif algorithm in ["ID3", "C4.5", "CART", "CHAID"] and root >= max_depth:
        final_decision = subdataset["Decision"].value_counts().idxmax()  # get the most frequent one
        terminateBuilding = True

    # -----------------------------------------------

    if enableParallelism == True:
        check_condition = "if"  # TODO: elif checks might be above than if statements in parallel
    else:
        if branch_index == 0:
            check_condition = "if"
        else:
            check_condition = "elif"

    check_rule = check_condition + " obj[" + str(winner_index) + "]" + compareTo + ":"

    leaf_id = str(uuid.uuid1())

    if enableParallelism != True:
        functions.storeRule(file, (functions.formatRule(root), "", check_rule))
    else:
        sample_rule = {}
        sample_rule["current_level"] = root
        sample_rule["leaf_id"] = leaf_id
        sample_rule["parents"] = parents
        sample_rule["rule"] = check_rule
        sample_rule["feature_idx"] = winner_index
        sample_rule["feature_name"] = winner_name
        sample_rule["instances"] = num_of_instances
        sample_rule["metric"] = metric
        sample_rule["return_statement"] = 0
        sample_rule["tree_id"] = tree_id

        # json to string
        sample_rule = json.dumps(sample_rule)

        custom_rules.append(sample_rule)

    # -----------------------------------------------

    if terminateBuilding == True:  # check decision is made
        parents = copy.copy(leaf_id)
        leaf_id = str(uuid.uuid1())

        decision_rule = "return " + charForResp + str(final_decision) + charForResp

        if enableParallelism != True:
            # serial
            functions.storeRule(file, (functions.formatRule(root + 1), decision_rule))
        else:
            # parallel
            sample_rule = {}
            sample_rule["current_level"] = root + 1
            sample_rule["leaf_id"] = leaf_id
            sample_rule["parents"] = parents
            sample_rule["rule"] = decision_rule
            sample_rule["feature_idx"] = winner_index
            sample_rule["feature_name"] = winner_name
            sample_rule["instances"] = num_of_instances
            sample_rule["metric"] = 0
            sample_rule["return_statement"] = 1
            sample_rule["tree_id"] = tree_id

            # json to string
            sample_rule = json.dumps(sample_rule)

            custom_rules.append(sample_rule)

    else:  # decision is not made, continue to create branch and leafs
        root = root + 1  # the following rule will be included by this rule. increase root
        parents = copy.copy(leaf_id)

        results = buildDecisionTree(
            subdataset,
            root,
            file,
            config,
            dataset_features,
            root - 1,
            leaf_id,
            parents,
            tree_id=tree_id,
            main_process_id=main_process_id,
        )

        custom_rules = custom_rules + results

        root = tmp_root * 1
        parents = copy.copy(parents_raw)

    gc.collect()

    return custom_rules


def buildDecisionTree(
    df: pd.DataFrame,
    root: int,
    file: str,
    config: dict,
    dataset_features: dict,
    parent_level: int = 0,
    leaf_id: int = 0,
    parents: str = "root",
    tree_id: int = 0,
    validation_df: Optional[pd.DataFrame] = None,
    main_process_id: Optional[int] = None,
) -> list:
    """
    Build a decition tree rules
    Args:
        df (pd.DataFrame): (sub) train set's data frame
        root (int): current depth
        file (str): target file to store rules
        config (dict): training configuration
        dataset_features (dict): pivot information about current
            dataframe's features with types
        parent_level (int): depth of the parent
        leaf_id (int): index of the current leaf. will use if
            for the first, and elif for the rest
        parents (str): id of the parent leaf as uuid. root is for
            the top one.
        tree_id (int): tree's id
        validation_df (pd.DataFrame): validation dataframe
        main_process_id (int): process id of the main trx
    Returns:
        models (list): list of dict as built model information
    """
    models = []

    decision_rules = []

    feature_names = df.columns[0:-1]

    enableParallelism = config["enableParallelism"]

    json_file = file.split(".")[0] + ".json"

    # --------------------------------------

    df_copy = df.copy()

    winner_name, num_of_instances, metric, _ = findDecision(df, config)

    # find winner index, this cannot be returned by find decision
    # because columns dropped in previous steps
    j = 0
    for i in dataset_features:
        if i == winner_name:
            winner_index = j
        j = j + 1

    numericColumn = False
    if dataset_features[winner_name] != "object":
        numericColumn = True

    # restoration
    columns = df.shape[1]
    for i in range(0, columns - 1):
        column_name = df_copy.columns[i]
        column_type = df_copy[column_name].dtypes
        if column_type != "object" and column_name != winner_name:
            df[column_name] = df_copy[column_name]

    classes = df[winner_name].value_counts().keys().tolist()
    logger.debug(f"classes: {classes} in {winner_name}")
    # -----------------------------------------------------

    num_cores = config["num_cores"]

    input_params = []

    # serial approach
    for i, current_class in enumerate(classes):
        subdataset = df[df[winner_name] == current_class]
        subdataset = subdataset.drop(columns=[winner_name])
        branch_index = i * 1

        # create branches serially
        if enableParallelism != True:
            if i == 0:
                descriptor = {
                    "feature": winner_name,
                    "instances": num_of_instances,
                    # "metric_name": metric_name,
                    "metric_value": round(metric, 4),
                    "depth": parent_level + 1,
                }
                descriptor = "# " + json.dumps(descriptor)

                functions.storeRule(file, (functions.formatRule(root), "", descriptor))

            results = createBranch(
                config,
                current_class,
                subdataset,
                numericColumn,
                branch_index,
                winner_name,
                winner_index,
                root,
                parents,
                file,
                dataset_features,
                num_of_instances,
                metric,
                tree_id=tree_id,
                main_process_id=main_process_id,
            )

            decision_rules = decision_rules + results

        else:
            input_params.append(
                (
                    config,
                    current_class,
                    subdataset,
                    numericColumn,
                    branch_index,
                    winner_name,
                    winner_index,
                    root,
                    parents,
                    file,
                    dataset_features,
                    num_of_instances,
                    metric,
                    tree_id,
                    main_process_id,
                )
            )

    # ---------------------------
    # add else condition in the decision tree
    if df.Decision.dtypes == "object":  # classification
        pivot = pd.DataFrame(df.Decision.value_counts()).sort_values(by=["count"], ascending=False)
        else_decision = f"return '{str(pivot.iloc[0].name)}'"

        if enableParallelism != True:
            functions.storeRule(file, (functions.formatRule(root), "else:"))
            functions.storeRule(file, (functions.formatRule(root + 1), else_decision))
        else:  # parallelism
            leaf_id = str(uuid.uuid1())

            check_rule = "else: " + else_decision

            sample_rule = {}
            sample_rule["current_level"] = root
            sample_rule["leaf_id"] = leaf_id
            sample_rule["parents"] = parents
            sample_rule["rule"] = check_rule
            sample_rule["feature_idx"] = -1
            sample_rule["feature_name"] = ""
            sample_rule["instances"] = df.shape[0]
            sample_rule["metric"] = 0
            sample_rule["return_statement"] = 0
            sample_rule["tree_id"] = tree_id

            # json to string
            sample_rule = json.dumps(sample_rule)
            decision_rules.append(sample_rule)

    else:  # regression
        else_decision = f"return {df.Decision.mean()}"

        if enableParallelism != True:
            functions.storeRule(file, (functions.formatRule(root), "else:"))
            functions.storeRule(file, (functions.formatRule(root + 1), else_decision))
        else:
            leaf_id = str(uuid.uuid1())

            check_rule = "else: " + else_decision

            sample_rule = {}
            sample_rule["current_level"] = root
            sample_rule["leaf_id"] = leaf_id
            sample_rule["parents"] = parents
            sample_rule["rule"] = check_rule
            sample_rule["tree_id"] = tree_id
            sample_rule["feature_name"] = ""
            sample_rule["instances"] = 0
            sample_rule["metric"] = 0
            sample_rule["return_statement"] = 1

            # json to string
            sample_rule = json.dumps(sample_rule)
            decision_rules.append(sample_rule)

    # ---------------------------

    try:
        main_process = psutil.Process(main_process_id)
        children = main_process.children(recursive=True)
        active_processes = len(children) + 1  # plus parent
        # active_processes = len(children)
    except:
        active_processes = 100  # set a large initial value

    results = []
    # create branches in parallel
    if enableParallelism == True:
        required_threads = active_processes + len(classes)

        # if parent_level == 0 and random_forest_enabled != True:
        if (
            main_process_id != None and num_cores >= required_threads
        ):  # len(classes) branches will be run in parallel
            # POOL_SIZE = num_cores
            POOL_SIZE = len(classes)

            # with closing(multiprocessing.Pool(POOL_SIZE)) as pool:
            with closing(CustomPool(POOL_SIZE)) as pool:
                funclist = []

                for input_param in input_params:
                    f = pool.apply_async(createBranchWrapper, [createBranch, input_param])
                    funclist.append(f)

                # all functions registered here

                for f in funclist:
                    branch_results = f.get(timeout=100000)

                    for branch_result in branch_results:
                        results.append(branch_result)

                pool.close()
                pool.terminate()

            # --------------------------------

        else:  # serial
            for input_param in input_params:
                sub_results = createBranchWrapper(createBranch, input_param)
                for sub_result in sub_results:
                    results.append(sub_result)

        # --------------------------------

        decision_rules = decision_rules + results

        # --------------------------------

        if root != 1:  # return children results until the root node
            return decision_rules

    # ---------------------------------------------

    if root == 1:
        if enableParallelism == True:
            # custom rules are stored in decision_rules. merge them all in a json file first

            json_rules = "[\n"  # initialize

            file_index = 0
            for custom_rule in decision_rules:
                json_rules += custom_rule

                if file_index < len(decision_rules) - 1:
                    json_rules += ", "

                json_rules += "\n"

                file_index = file_index + 1

            # -----------------------------------

            json_rules += "]"
            functions.createFile(json_file, json_rules)

            # -----------------------------------
            # reconstruct rules from json to py

            reconstructRules(json_file, feature_names)

            # -----------------------------------

        # is regular decision tree
        if (
            config["enableRandomForest"] != True
            and config["enableGBM"] != True
            and config["enableAdaboost"] != True
        ):
            # this is reguler decision tree. find accuracy here.

            module_name = "outputs.rules.rules"
            myrules = load_module(module_name)  # rules0
            models.append(myrules)

    return models


def reconstructRules(source: str, feature_names, tree_id: int = 0) -> None:
    """
    Reconstruct rules as python files from randomly created json
        files in parallel runs
    Args:
        source (str): json file name (e.g. outputs/rules/rules.json)
        feature_names (list): feature names
        tree_id (int): tree's id
    Returns:
        None
    """
    logger.debug(f"Reconstructing {source}")

    file_name = source.split(".json")[0]
    file_name = file_name + ".py"

    # -----------------------------------

    constructor = "def findDecision(obj): #"
    idx = 0
    for feature in feature_names:
        constructor = constructor + "obj[" + str(idx) + "]: " + feature

        if idx < len(feature_names) - 1:
            constructor = constructor + ", "
        idx = idx + 1

    functions.createFile(file_name, constructor + "\n")

    # -----------------------------------

    with open(source, "r", encoding="UTF-8") as f:
        rules = json.load(f)

    logger.debug(rules)

    def padleft(rule, level):
        for _ in range(0, level):
            rule = "\t" + rule
        return rule

    logger.debug("def findDecision(obj):")

    rule_set = []
    # json file might not store rules respectively
    for instance in rules:
        if len(instance) > 0:
            rule = []
            rule.append(instance["current_level"])
            rule.append(instance["leaf_id"])
            rule.append(instance["parents"])
            rule.append(instance["rule"])
            rule.append(instance["feature_name"])
            rule.append(instance["instances"])
            rule.append(instance["metric"])
            rule.append(instance["return_statement"])
            rule_set.append(rule)
            logger.debug(padleft(instance["rule"], instance["current_level"]))

    np_rules = np.array(rule_set)

    def extract_rules(np_rules: np.ndarray, parent: str = "root", level: int = 1) -> None:
        """
        Extract rules
        Args:
            np_rules (ndarray): (sub) training data frame
            parent (str): parent leaf's uuid or root for the top one
            level (int): current depth
        Returns:
            None
        """
        level_raw = level * 1
        parent_raw = copy.copy(parent)

        else_rule = ""

        leaf_idx = 0
        for i in range(0, np_rules.shape[0]):
            current_level = int(np_rules[i][0])
            leaf_id = np_rules[i][1]
            parent_id = np_rules[i][2]
            rule = np_rules[i][3]
            feature_name = np_rules[i][4]
            instances = int(np_rules[i][5])
            metric = float(np_rules[i][6])
            return_statement = int(np_rules[i][7])

            if parent_id == parent:
                if_statement = False
                if rule[0:2] == "if":
                    if_statement = True

                else_statement = False
                if rule[0:5] == "else:":
                    else_statement = True
                    else_rule = rule

                # ------------------------

                if else_statement != True:
                    if if_statement == True and leaf_idx > 0:
                        rule = "el" + rule

                    logger.debug(f"{padleft(rule, level)} ({leaf_idx})")

                    if leaf_idx == 0 and return_statement == 0:
                        explainer = {}
                        explainer["feature"] = feature_name
                        explainer["instances"] = instances
                        explainer["metric_value"] = round(metric, 4)
                        explainer["depth"] = current_level
                        explainer = "# " + json.dumps(explainer)
                        functions.storeRule(file_name, padleft(explainer, level))

                    functions.storeRule(file_name, padleft(rule, level))

                    level = level + 1
                    parent = copy.copy(leaf_id)
                    extract_rules(np_rules, parent, level)
                    level = level_raw * 1
                    parent = copy.copy(parent_raw)  # restore

                    leaf_idx = leaf_idx + 1

        # add else statement

        if else_rule != "":
            logger.debug(padleft(else_rule, level))
            functions.storeRule(file_name, padleft(else_rule, level))

    # end of extract_rules
    # ------------------------------------

    extract_rules(np_rules)
