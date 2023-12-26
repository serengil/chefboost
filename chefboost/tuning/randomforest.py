from typing import Optional
import multiprocessing
from contextlib import closing

from tqdm import tqdm
import pandas as pd

from chefboost.commons import functions
from chefboost.training import Training
from chefboost.commons.module import load_module

# pylint: disable=unused-argument


def apply(
    df: pd.DataFrame,
    config: dict,
    header: str,
    dataset_features: dict,
    validation_df: Optional[pd.DataFrame] = None,
    process_id: Optional[int] = None,
    silent: bool = False,
) -> list:
    """
    Train procedure of random forest
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

    num_of_trees = config["num_of_trees"]

    parallelism_on = config["enableParallelism"]

    # TODO: is this logical for 48x2 cores?
    # config["enableParallelism"] = False #run each tree in parallel but each branch in serial

    # TODO: reconstruct for parallel run is problematic. you should reconstruct based on tree id.

    input_params = []

    pbar = tqdm(range(0, num_of_trees), desc="Bagging", disable=silent)
    for i in pbar:
        if silent is False:
            pbar.set_description(f"Sub decision tree {i + 1} is processing")
        subset = df.sample(frac=1 / num_of_trees)

        root = 1

        module_name = "outputs/rules/rule_" + str(i)
        file = module_name + ".py"

        functions.createFile(file, header)

        if parallelism_on:  # parallel run
            input_params.append(
                (
                    subset,
                    root,
                    file,
                    config,
                    dataset_features,
                    0,
                    0,
                    "root",
                    i,
                    None,
                    process_id,
                )
            )

        else:  # serial run
            Training.buildDecisionTree(
                subset,
                root,
                file,
                config,
                dataset_features,
                parent_level=0,
                leaf_id=0,
                parents="root",
                tree_id=i,
                main_process_id=process_id,
            )

    # -------------------------------

    if parallelism_on:
        num_cores = config["num_cores"]

        # ---------------------------------

        if num_of_trees <= num_cores:
            POOL_SIZE = num_of_trees
        else:
            POOL_SIZE = num_cores

        with closing(multiprocessing.Pool(POOL_SIZE)) as pool:
            funclist = []
            for input_param in input_params:
                f = pool.apply_async(buildDecisionTree, [*input_param])
                funclist.append(f)

            # all functions registered here
            # results = []
            for f in tqdm(funclist, disable=silent):
                _ = f.get(timeout=100000)  # this was branch_results
                # results.append(branch_results)

            pool.close()
            pool.terminate()

    # -------------------------------
    # collect models for both serial and parallel here
    for i in range(0, num_of_trees):
        module_name = "outputs/rules/rule_" + str(i)
        myrules = load_module(module_name)
        models.append(myrules)

    # -------------------------------

    return models


# wrapper for parallel run
def buildDecisionTree(
    df,
    root,
    file,
    config,
    dataset_features,
    parent_level,
    leaf_id,
    parents,
    tree_id,
    validation_df=None,
    process_id=None,
):
    """
    Wrapper for Training.buildDecisionTree
    """
    Training.buildDecisionTree(
        df,
        root,
        file,
        config,
        dataset_features,
        parent_level=parent_level,
        leaf_id=leaf_id,
        parents=parents,
        tree_id=tree_id,
        main_process_id=process_id,
    )
