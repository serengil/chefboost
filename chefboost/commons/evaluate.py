import math
import pandas as pd
from chefboost.commons.logger import Logger

# pylint: disable=broad-except

logger = Logger(module="chefboost/commons/evaluate.py")


def evaluate(df: pd.DataFrame, task: str = "train", silent: bool = False) -> dict:
    """
    Evaluate results
    Args:
        df (pd.DataFrame): data frame
        task (str): train, test
        silent (bool): set this to True if you do not want to
            see any informative logs
    Returns:
        evaluation results (dict)
    """
    if df["Decision"].dtypes == "object":
        problem_type = "classification"
    else:
        problem_type = "regression"

    evaluation_results = {}
    instances = df.shape[0]

    if silent is False:
        logger.info("-------------------------")
        logger.info(f"Evaluate {task} set")
        logger.info("-------------------------")

    if problem_type == "classification":
        idx = df[df["Prediction"] == df["Decision"]].index
        accuracy = 100 * len(idx) / df.shape[0]
        if silent is False:
            logger.info(f"Accuracy: {accuracy}% on {instances} instances")

        evaluation_results["Accuracy"] = accuracy
        evaluation_results["Instances"] = instances
        # -----------------------------

        predictions = df.Prediction.values
        actuals = df.Decision.values

        # -----------------------------
        # confusion matrix

        # labels = df.Prediction.unique()
        labels = df.Decision.unique()

        confusion_matrix = []
        for prediction_label in labels:
            confusion_row = []
            for actual_label in labels:
                item = len(
                    df[(df["Prediction"] == prediction_label) & (df["Decision"] == actual_label)][
                        "Decision"
                    ].values
                )
                confusion_row.append(item)
            confusion_matrix.append(confusion_row)

        if silent is False:
            logger.info(f"Labels: {labels}")
            logger.info(f"Confusion matrix: {confusion_matrix}")

        evaluation_results["Labels"] = labels
        evaluation_results["Confusion matrix"] = confusion_matrix

        # -----------------------------
        # precision and recall

        for decision_class in labels:
            fp = 0
            fn = 0
            tp = 0
            tn = 0
            for i, prediction in enumerate(predictions):
                actual = actuals[i]

                if actual == decision_class and prediction == decision_class:
                    tp = tp + 1
                # pylint: disable=consider-using-in
                elif actual != decision_class and prediction != decision_class:
                    tn = tn + 1
                elif actual != decision_class and prediction == decision_class:
                    fp = fp + 1
                elif actual == decision_class and prediction != decision_class:
                    fn = fn + 1

            epsilon = 0.0000001  # to avoid divison by zero exception
            precision = round(100 * tp / (tp + fp + epsilon), 4)
            recall = round(100 * tp / (tp + fn + epsilon), 4)  # tpr
            f1_score = round((2 * precision * recall) / (precision + recall + epsilon), 4)
            accuracy = round(100 * (tp + tn) / (tp + tn + fp + fn + epsilon), 4)

            if len(labels) >= 3:
                if silent is False:
                    logger.info(f"Decision {decision_class}")
                    logger.info(f"Accuracy: {accuracy}")

                evaluation_results[f"Decision {decision_class}'s Accuracy"] = accuracy

            if silent is False:
                logger.info(f"Precision: {precision}%, Recall: {recall}%, F1: {f1_score}%")
                logger.debug(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

            evaluation_results["Precision"] = precision
            evaluation_results["Recall"] = recall
            evaluation_results["F1"] = f1_score

            if len(labels) < 3:
                break

    # -------------------------------------
    else:
        df["Absolute_Error"] = abs(df["Prediction"] - df["Decision"])
        df["Absolute_Error_Squared"] = df["Absolute_Error"] * df["Absolute_Error"]
        df["Decision_Squared"] = df["Decision"] * df["Decision"]
        df["Decision_Mean"] = df["Decision"].mean()

        logger.debug(df)

        if instances > 0:
            mae = df["Absolute_Error"].sum() / instances
            mse = df["Absolute_Error_Squared"].sum() / instances
            rmse = math.sqrt(mse)

            evaluation_results["MAE"] = mae
            evaluation_results["MSE"] = mse
            evaluation_results["RMSE"] = rmse

            if silent is False:
                logger.info(f"MAE: {mae}")
                logger.info(f"MSE: {mse}")
                logger.info(f"RMSE: {rmse}")

            rae = 0
            rrse = 0
            try:  # divisor might be equal to 0.
                rae = math.sqrt(df["Absolute_Error_Squared"].sum()) / math.sqrt(
                    df["Decision_Squared"].sum()
                )

                rrse = math.sqrt(
                    (df["Absolute_Error_Squared"].sum())
                    / ((df["Decision_Mean"] - df["Decision"]) ** 2).sum()
                )

            except Exception as err:
                logger.error(str(err))

            if silent is False:
                logger.info(f"RAE: {rae}")
                logger.info(f"RRSE {rrse}")

            evaluation_results["RAE"] = rae
            evaluation_results["RRSE"] = rrse

            mean = df["Decision"].mean()

            if silent is False:
                logger.info(f"Mean: {mean}")

            evaluation_results["Mean"] = mean

            if mean > 0:
                if silent is False:
                    logger.info(f"MAE / Mean: {100 * mae / mean}%")
                    logger.info(f"RMSE / Mean: {100 * rmse / mean}%")

                evaluation_results["MAE / Mean"] = 100 * mae / mean
                evaluation_results["RMSE / Mean"] = 100 * rmse / mean

    return evaluation_results
