"""
The script for generating DSM2 salinity dataset.

"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsdb import pickle_dump

sys.path.append("..")
from modeling.utils import setup_logger
from dataset_generating_scripts.data_processing_utils import (
    window_truncate,
    add_artificial_mask,
    saving_into_h5,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DSM2 salinity dataset")
    parser.add_argument(
        "--file_path", 
        help="path of dataset file", 
        type=str
    )
    parser.add_argument(
        "--mask_type", 
        help="type of forced missing data mask to use", 
        type=str,
        default="sparse"
    )
    parser.add_argument(
        "--miss_percent",
        help="percent of missing data in mask. applies only for sparse mask type.",
        type=float,
        default=20,
    )
    parser.add_argument(
        "--seq_len", 
        help="sequence length", 
        type=int, 
        default=100
    )
    parser.add_argument(
        "--mask_features_num", 
        help="number of features to mask. only applies for block mask type.", 
        type=int, 
        default=1
    )
    parser.add_argument(
        "--mask_block_len", 
        help="block missing length. only applies for block mask type.", 
        type=int, 
        default=2
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
        "Generate DSM2 salinity dataset",
        mode="w",
    )
    logger.info(args)

    # set rng seed for repeatability between runs
    np.random.seed(0)

    # read in data
    df_collector = []
    station_name_collector = []
    file_list = os.listdir(args.file_path)
    for filename in file_list:
        file_path = os.path.join(args.file_path, filename)
        current_df = pd.read_excel(file_path)
        df_collector.append(current_df)
        logger.info(f"reading {file_path}, data shape {current_df.shape}")
        station_name_collector += (current_df.columns[1:].values.tolist())

    logger.info(
        f"There are total {len(station_name_collector)} stations, they are {station_name_collector}"
    )
    date_time = df_collector[0]["Time"]
    df_collector = [i.drop("Time", axis=1) for i in df_collector]
    df = pd.concat(df_collector, axis=1)

    # use only important stations
    station_names = ['RSAC064', 'RSAC075', 'RSAC081', 'RSAC092', 'RSAN007','RSAN018']
    df = df[:][station_names].reset_index()
    args.feature_names = station_names
    args.feature_num = len(station_names)

    # missing data is marked with -2 values, replace -2 with NaN
    df[df.eq(-2)] = np.NaN

    logger.info(
        f"Original df missing rate: "
        f"{(df[args.feature_names].isna().sum().sum() / (df.shape[0] * args.feature_num)):.3f}"
    )

    # drop samples with missing data
    seq_len = args.seq_len
    num_stations = args.feature_num
    num_samples = int(np.floor(len(df)/seq_len))
    data = df.to_numpy()[:num_samples*seq_len,1:].reshape(num_samples,seq_len,num_stations)
    data_collector = []
    for sample in range(num_samples):
        if ~np.isnan(data[sample]).any():
            data_collector.append(data[sample])
    data_filtered = np.asanyarray(data_collector).reshape(-1,num_stations)
    df = pd.DataFrame(data=data_filtered, columns=station_names)
    data_len = df.shape[0]

    # separate data in to training, validation, and testing sets
    # use 60% of data for training, 20% for validation, and 20% for testing
    selected_as_train = np.arange(0, int(np.round(data_len*0.6)))
    selected_as_val = np.arange(int(np.round(data_len*0.6)), int(np.round(data_len*0.8)))
    selected_as_test = np.arange(int(np.round(data_len*0.8)), data_len)
    #df["date_time"] = date_time
    train_set = df[df.index.isin(selected_as_train)]
    val_set = df[df.index.isin(selected_as_val)]
    test_set = df[df.index.isin(selected_as_test)]

    # Normalize the data
    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, args.feature_names])
    val_set_X = scaler.transform(val_set.loc[:, args.feature_names])
    test_set_X = scaler.transform(test_set.loc[:, args.feature_names])

    train_set_X = window_truncate(train_set_X, args.seq_len)
    val_set_X = window_truncate(val_set_X, args.seq_len)
    test_set_X = window_truncate(test_set_X, args.seq_len)

    train_set_dict = add_artificial_mask(train_set_X, args.miss_percent/100, "train", args.mask_features_num, 
        args.mask_block_len, args.mask_type)
    val_set_dict = add_artificial_mask(val_set_X, args.miss_percent/100, "val", args.mask_features_num, 
        args.mask_block_len, args.mask_type)
    test_set_dict = add_artificial_mask(test_set_X, args.miss_percent/100, "test", args.mask_features_num, 
        args.mask_block_len, args.mask_type)
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

    logger.info(
        f"Feature num: {args.feature_num},\n"
        f'Sample num in train set: {len(train_set_dict["X"])}\n'
        f'Sample num in val set: {len(val_set_dict["X"])}\n'
        f'Sample num in test set: {len(test_set_dict["X"])}\n'
    )

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)
    pickle_dump(scaler, os.path.join(dataset_saving_dir, 'scaler'))
    logger.info(f"All done. Saved to {dataset_saving_dir}.")
