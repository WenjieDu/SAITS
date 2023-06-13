"""
Generating datasets used in the NRTSI paper for comparison.

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


# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import argparse
import os
import sys

import numpy as np

sys.path.append("..")
from modeling.utils import setup_logger
from dataset_generating_scripts.data_processing_utils import (
    add_artificial_mask,
    saving_into_h5,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets from NRTSI")
    parser.add_argument("--file_path", help="path of dataset file", type=str)
    parser.add_argument(
        "--artificial_missing_rate",
        help="artificially mask out additional values",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--dataset_name",
        help="name of generated dataset, will be the name of saving dir",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--saving_path", type=str, help="parent dir of generated dataset", default="."
    )
    args = parser.parse_args()

    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    logger = setup_logger(
        os.path.join(dataset_saving_dir + "/dataset_generating.log"),
        "Generate NRTSI dataset",
        mode="w",
    )
    logger.info(args)

    train_set_X = np.load(os.path.join(args.file_path, "train.npy"))
    val_set_X = np.load(os.path.join(args.file_path, "val.npy"))
    test_set_X = np.load(os.path.join(args.file_path, "test.npy"))

    train_set_dict = add_artificial_mask(
        train_set_X, args.artificial_missing_rate, "train"
    )
    val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, "val")
    test_set_dict = add_artificial_mask(
        test_set_X, args.artificial_missing_rate, "test"
    )

    logger.info(
        f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}'
    )
    logger.info(
        f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}'
    )

    processed_data = {
        "train": train_set_dict,
        "val": val_set_dict,
        "test": test_set_dict,
    }

    args.feature_num = train_set_X.shape[-1]
    logger.info(
        f"Feature num: {args.feature_num},\n"
        f'Sample num in train set: {len(train_set_dict["X"])}\n'
        f'Sample num in val set: {len(val_set_dict["X"])}\n'
        f'Sample num in test set: {len(test_set_dict["X"])}\n'
    )

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)
    logger.info(f"All done. Saved to {dataset_saving_dir}.")
