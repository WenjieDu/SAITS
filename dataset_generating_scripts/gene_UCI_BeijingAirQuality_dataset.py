import argparse
import os
import sys

import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append('..')
from modeling.utils import setup_logger
from dataset_generating_scripts.data_processing_utils import window_truncate, add_artificial_mask, saving_into_h5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate UCI air quality dataset')
    parser.add_argument("--file_path", help='path of dataset file', type=str)
    parser.add_argument("--artificial_missing_rate", help='artificially mask out additional values',
                        type=float, default=0.1)
    parser.add_argument("--seq_len", help='sequence length', type=int, default=100)
    parser.add_argument('--dataset_name', help='name of generated dataset, will be the name of saving dir', type=str,
                        default='test')
    parser.add_argument('--saving_path', type=str, help='parent dir of generated dataset', default='.')
    args = parser.parse_args()

    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    logger = setup_logger(os.path.join(dataset_saving_dir + "/dataset_generating.log"),
                          'Generate UCI air quality dataset', mode='w')
    logger.info(args)

    df_collector = []
    station_name_collector = []
    file_list = os.listdir(args.file_path)
    for filename in file_list:
        file_path = os.path.join(args.file_path, filename)
        current_df = pd.read_csv(file_path)
        current_df['date_time'] = pd.to_datetime(current_df[['year', 'month', 'day', 'hour']])
        station_name_collector.append(current_df.loc[0, 'station'])
        # remove duplicated date info and wind direction, which is a categorical col
        current_df = current_df.drop(['year', 'month', 'day', 'hour', 'wd', 'No', 'station'], axis=1)
        df_collector.append(current_df)
        logger.info(f'reading {file_path}, data shape {current_df.shape}')

    logger.info(f'There are total {len(station_name_collector)} stations, they are {station_name_collector}')
    date_time = df_collector[0]['date_time']
    df_collector = [i.drop('date_time', axis=1) for i in df_collector]
    df = pd.concat(df_collector, axis=1)
    args.feature_names = [station + '_' + feature
                          for station in station_name_collector
                          for feature in df_collector[0].columns]
    args.feature_num = len(args.feature_names)
    df.columns = args.feature_names
    logger.info(f'Original df missing rate: '
                f'{(df[args.feature_names].isna().sum().sum() / (df.shape[0] * args.feature_num)):.3f}')

    df['date_time'] = date_time
    unique_months = df['date_time'].dt.to_period('M').unique()
    selected_as_test = unique_months[:10]  # select first 3 months as test set
    logger.info(f'months selected as test set are {selected_as_test}')
    selected_as_val = unique_months[10:20]  # select the 4th - the 6th months as val set
    logger.info(f'months selected as val set are {selected_as_val}')
    selected_as_train = unique_months[20:]  # use left months as train set
    logger.info(f'months selected as train set are {selected_as_train}')
    test_set = df[df['date_time'].dt.to_period('M').isin(selected_as_test)]
    val_set = df[df['date_time'].dt.to_period('M').isin(selected_as_val)]
    train_set = df[df['date_time'].dt.to_period('M').isin(selected_as_train)]

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, args.feature_names])
    val_set_X = scaler.transform(val_set.loc[:, args.feature_names])
    test_set_X = scaler.transform(test_set.loc[:, args.feature_names])

    train_set_X = window_truncate(train_set_X, args.seq_len)
    val_set_X = window_truncate(val_set_X, args.seq_len)
    test_set_X = window_truncate(test_set_X, args.seq_len)

    train_set_dict = add_artificial_mask(train_set_X, args.artificial_missing_rate, 'train')
    val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, 'val')
    test_set_dict = add_artificial_mask(test_set_X, args.artificial_missing_rate, 'test')
    logger.info(f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}')
    logger.info(f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}')

    processed_data = {
        'train': train_set_dict,
        'val': val_set_dict,
        'test': test_set_dict
    }

    logger.info(f'Feature num: {args.feature_num},\n'
                f'Sample num in train set: {len(train_set_dict["X"])}\n'
                f'Sample num in val set: {len(val_set_dict["X"])}\n'
                f'Sample num in test set: {len(test_set_dict["X"])}\n')

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)
    logger.info(f'All done. Saved to {dataset_saving_dir}.')
