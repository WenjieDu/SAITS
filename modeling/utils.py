"""
Utility functions are stored here.

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics

plt.rcParams["savefig.dpi"] = 300  # pixel
plt.rcParams["figure.dpi"] = 300  # resolution
plt.rcParams["figure.figsize"] = [8, 4]  # figure size


def masked_mae_cal(inputs, target, mask):
    """calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_mse_cal(inputs, target, mask):
    """calculate Mean Square Error"""
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_rmse_cal(inputs, target, mask):
    """calculate Root Mean Square Error"""
    return torch.sqrt(masked_mse_cal(inputs, target, mask))


def masked_mre_cal(inputs, target, mask):
    """calculate Mean Relative Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (
        torch.sum(torch.abs(target * mask)) + 1e-9
    )


def precision_recall(y_pred, y_test):
    precisions, recalls, thresholds = metrics.precision_recall_curve(
        y_true=y_test, probas_pred=y_pred
    )
    area = metrics.auc(recalls, precisions)
    return area, precisions, recalls, thresholds


def auc_roc(y_pred, y_test):
    auc = metrics.roc_auc_score(y_true=y_test, y_score=y_pred)
    fprs, tprs, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_pred)
    return auc, fprs, tprs, thresholds


def auc_to_recall(recalls, precisions, recall=0.01):
    precisions_mod = precisions.copy()
    ind = np.where(recalls < recall)[0][0] + 1
    precisions_mod[:ind] = 0
    area = metrics.auc(recalls, precisions_mod)
    return area


def cal_classification_metrics(probabilities, labels, pos_label=1, class_num=1):
    """
    pos_label: The label of the positive class.
    """
    if class_num == 1:
        class_predictions = (probabilities >= 0.5).astype(int)
    elif class_num == 2:
        class_predictions = np.argmax(probabilities, axis=1)
    else:
        assert "args.class_num>2, class need to be specified for precision_recall_fscore_support"
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        labels, class_predictions, pos_label=pos_label, warn_for=()
    )
    precision, recall, f1 = precision[1], recall[1], f1[1]
    precisions, recalls, _ = metrics.precision_recall_curve(
        labels, probabilities[:, -1], pos_label=pos_label
    )
    acc_score = metrics.accuracy_score(labels, class_predictions)
    ROC_AUC, fprs, tprs, thresholds = auc_roc(probabilities[:, -1], labels)
    PR_AUC = metrics.auc(recalls, precisions)
    classification_metrics = {
        "classification_predictions": class_predictions,
        "acc_score": acc_score,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precisions": precisions,
        "recalls": recalls,
        "fprs": fprs,
        "tprs": tprs,
        "ROC_AUC": ROC_AUC,
        "PR_AUC": PR_AUC,
    }
    return classification_metrics


def plot_AUCs(
    pdf_file, x_values, y_values, auc_value, title, x_name, y_name, dataset_name
):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        x_values,
        y_values,
        ".",
        label=f"{dataset_name}, AUC={auc_value:.3f}",
        rasterized=True,
    )
    l = ax.legend(fontsize=10, loc="lower left")
    l.set_zorder(20)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title, fontsize=12)
    pdf_file.savefig(fig)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise TypeError("Boolean value expected.")


def setup_logger(log_file_path, log_name, mode="a"):
    """set up log file
    mode : 'a'/'w' mean append/overwrite,
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False  # prevent the child logger from propagating log to the root logger (twice), not necessary
    return logger


class Controller:
    def __init__(self, early_stop_patience):
        self.original_early_stop_patience_value = early_stop_patience
        self.early_stop_patience = early_stop_patience
        self.state_dict = {
            # `step` is for training stage
            "train_step": 0,
            # below are for validation stage
            "val_step": 0,
            "epoch": 0,
            "best_imputation_MAE": 1e9,
            "should_stop": False,
            "save_model": False,
        }

    def epoch_num_plus_1(self):
        self.state_dict["epoch"] += 1

    def __call__(self, stage, info=None, logger=None):
        if stage == "train":
            self.state_dict["train_step"] += 1
        else:
            self.state_dict["val_step"] += 1
            self.state_dict["save_model"] = False
            current_imputation_MAE = info["imputation_MAE"]
            imputation_MAE_dropped = False  # flags to decrease early stopping patience

            # update best_loss
            if current_imputation_MAE < self.state_dict["best_imputation_MAE"]:
                logger.info(
                    f"best_imputation_MAE has been updated to {current_imputation_MAE}"
                )
                self.state_dict["best_imputation_MAE"] = current_imputation_MAE
                imputation_MAE_dropped = True
            if imputation_MAE_dropped:
                self.state_dict["save_model"] = True

            if self.state_dict["save_model"]:
                self.early_stop_patience = self.original_early_stop_patience_value
            else:
                # if use early_stopping, then update its patience
                if self.early_stop_patience > 0:
                    self.early_stop_patience -= 1
                elif self.early_stop_patience == 0:
                    logger.info(
                        "early_stop_patience has been exhausted, stop training now"
                    )
                    self.state_dict["should_stop"] = True  # to stop training process
                else:
                    pass  # which means early_stop_patience_value is set as -1, not work

        return self.state_dict


def check_saving_dir_for_model(args, time_now):
    saving_path = os.path.join(args.result_saving_base_dir, args.model_name)
    if not args.test_mode:
        log_saving = os.path.join(saving_path, "logs")
        model_saving = os.path.join(saving_path, "models")
        sub_model_saving = os.path.join(model_saving, time_now)
        [
            os.makedirs(dir_)
            for dir_ in [model_saving, log_saving, sub_model_saving]
            if not os.path.exists(dir_)
        ]
        return sub_model_saving, log_saving
    else:
        log_saving = os.path.join(saving_path, "test_log")
        return None, log_saving


def save_model(model, optimizer, model_state_info, args, saving_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(), # don't save optimizer, considering GANs have 2 optimizers
        "training_step": model_state_info["train_step"],
        "epoch": model_state_info["epoch"],
        "model_state_info": model_state_info,
        "args": args,
    }
    torch.save(checkpoint, saving_path)


def load_model(model, checkpoint_path, logger):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Already restored model from checkpoint: {checkpoint_path}")
    return model


def load_model_saved_with_module(model, checkpoint_path, logger):
    """
    To load models those are trained in parallel and saved with module (need to remove 'module.'
    """
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = dict()
    for k, v in checkpoint["model_state_dict"].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    logger.info(f"Already restored model from checkpoint: {checkpoint_path}")
    return model
