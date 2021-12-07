import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def parse_delta(masks, seq_len, feature_num):
    """generate deltas from masks, used in BRITS"""
    deltas = []
    for h in range(seq_len):
        if h == 0:
            deltas.append(np.zeros(feature_num))
        else:
            deltas.append(np.ones(feature_num) + (1 - masks[h]) * deltas[-1])
    return np.asarray(deltas)


def fill_with_last_observation(arr):
    """ namely forward-fill nan values
    https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    out = np.nan_to_num(out)  # if nan still exists then fill with 0
    return out


class LoadDataset(Dataset):
    def __init__(self, file_path, seq_len, feature_num, model_type):
        super(LoadDataset, self).__init__()
        self.file_path = file_path
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.model_type = model_type


class LoadValTestDataset(LoadDataset):
    """Loading process of val or test set"""

    def __init__(self, file_path, set_name, seq_len, feature_num, model_type):
        super(LoadValTestDataset, self).__init__(file_path, seq_len, feature_num, model_type)
        with h5py.File(self.file_path, 'r') as hf:  # read data from h5 file
            self.X = hf[set_name]['X'][:]
            self.X_hat = hf[set_name]['X_hat'][:]
            self.missing_mask = hf[set_name]['missing_mask'][:]
            self.indicating_mask = hf[set_name]['indicating_mask'][:]

        # fill missing values with 0
        self.X = np.nan_to_num(self.X)
        self.X_hat = np.nan_to_num(self.X_hat)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.model_type in ['Transformer', 'SAITS']:
            sample = (
                torch.tensor(idx),
                torch.from_numpy(self.X_hat[idx].astype('float32')),
                torch.from_numpy(self.missing_mask[idx].astype('float32')),
                torch.from_numpy(self.X[idx].astype('float32')),
                torch.from_numpy(self.indicating_mask[idx].astype('float32')),
            )
        elif self.model_type in ['BRITS', 'MRNN']:
            forward = {'X_hat': self.X_hat[idx], 'missing_mask': self.missing_mask[idx],
                       'deltas': parse_delta(self.missing_mask[idx], self.seq_len, self.feature_num)}
            backward = {'X_hat': np.flip(forward['X_hat'], axis=0).copy(),
                        'missing_mask': np.flip(forward['missing_mask'], axis=0).copy()}
            backward['deltas'] = parse_delta(backward['missing_mask'], self.seq_len, self.feature_num)
            sample = (
                torch.tensor(idx),
                # for forward
                torch.from_numpy(forward['X_hat'].astype('float32')),
                torch.from_numpy(forward['missing_mask'].astype('float32')),
                torch.from_numpy(forward['deltas'].astype('float32')),
                # for backward
                torch.from_numpy(backward['X_hat'].astype('float32')),
                torch.from_numpy(backward['missing_mask'].astype('float32')),
                torch.from_numpy(backward['deltas'].astype('float32')),

                torch.from_numpy(self.X[idx].astype('float32')),
                torch.from_numpy(self.indicating_mask[idx].astype('float32')),
            )
        else:
            assert ValueError, f'Error model type: {self.model_type}'
        return sample


class LoadTrainDataset(LoadDataset):
    """Loading process of train set"""

    def __init__(self, file_path, seq_len, feature_num, model_type, masked_imputation_task):
        super(LoadTrainDataset, self).__init__(file_path, seq_len, feature_num, model_type)
        self.masked_imputation_task = masked_imputation_task
        if masked_imputation_task:
            self.artificial_missing_rate = 0.2
            assert 0 < self.artificial_missing_rate < 1, 'artificial_missing_rate should be greater than 0 and less than 1'

        with h5py.File(self.file_path, 'r') as hf:  # read data from h5 file
            self.X = hf['train']['X'][:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        if self.masked_imputation_task:
            X = X.reshape(-1)
            indices = np.where(~np.isnan(X))[0].tolist()
            indices = np.random.choice(indices, round(len(indices) * self.artificial_missing_rate))
            X_hat = np.copy(X)
            X_hat[indices] = np.nan  # mask values selected by indices
            missing_mask = (~np.isnan(X_hat)).astype(np.float32)
            indicating_mask = ((~np.isnan(X)) ^ (~np.isnan(X_hat))).astype(np.float32)
            X = np.nan_to_num(X)
            X_hat = np.nan_to_num(X_hat)
            # reshape into time series
            X = X.reshape(self.seq_len, self.feature_num)
            X_hat = X_hat.reshape(self.seq_len, self.feature_num)
            missing_mask = missing_mask.reshape(self.seq_len, self.feature_num)
            indicating_mask = indicating_mask.reshape(self.seq_len, self.feature_num)

            if self.model_type in ['Transformer', 'SAITS']:
                sample = (
                    torch.tensor(idx),
                    torch.from_numpy(X_hat.astype('float32')),
                    torch.from_numpy(missing_mask.astype('float32')),
                    torch.from_numpy(X.astype('float32')),
                    torch.from_numpy(indicating_mask.astype('float32')),
                )
            elif self.model_type in ['BRITS', 'MRNN']:
                forward = {'X_hat': X_hat, 'missing_mask': missing_mask,
                           'deltas': parse_delta(missing_mask, self.seq_len, self.feature_num)}

                backward = {'X_hat': np.flip(forward['X_hat'], axis=0).copy(),
                            'missing_mask': np.flip(forward['missing_mask'], axis=0).copy()}
                backward['deltas'] = parse_delta(backward['missing_mask'], self.seq_len, self.feature_num)
                sample = (
                    torch.tensor(idx),
                    # for forward
                    torch.from_numpy(forward['X_hat'].astype('float32')),
                    torch.from_numpy(forward['missing_mask'].astype('float32')),
                    torch.from_numpy(forward['deltas'].astype('float32')),
                    # for backward
                    torch.from_numpy(backward['X_hat'].astype('float32')),
                    torch.from_numpy(backward['missing_mask'].astype('float32')),
                    torch.from_numpy(backward['deltas'].astype('float32')),

                    torch.from_numpy(X.astype('float32')),
                    torch.from_numpy(indicating_mask.astype('float32')),
                )
            else:
                assert ValueError, f'Error model type: {self.model_type}'
        else:
            # if training without masked imputation task, then there is no need to artificially mask out observed values
            missing_mask = (~np.isnan(X)).astype(np.float32)
            X = np.nan_to_num(X)
            if self.model_type in ['Transformer', 'SAITS']:
                sample = (
                    torch.tensor(idx),
                    torch.from_numpy(X.astype('float32')),
                    torch.from_numpy(missing_mask.astype('float32')),
                )
            elif self.model_type in ['BRITS', 'MRNN']:
                forward = {'X': X, 'missing_mask': missing_mask,
                           'deltas': parse_delta(missing_mask, self.seq_len, self.feature_num)}
                backward = {'X': np.flip(forward['X'], axis=0).copy(),
                            'missing_mask': np.flip(forward['missing_mask'], axis=0).copy()}
                backward['deltas'] = parse_delta(backward['missing_mask'], self.seq_len, self.feature_num)
                sample = (
                    torch.tensor(idx),
                    # for forward
                    torch.from_numpy(forward['X'].astype('float32')),
                    torch.from_numpy(forward['missing_mask'].astype('float32')),
                    torch.from_numpy(forward['deltas'].astype('float32')),
                    # for backward
                    torch.from_numpy(backward['X'].astype('float32')),
                    torch.from_numpy(backward['missing_mask'].astype('float32')),
                    torch.from_numpy(backward['deltas'].astype('float32')),
                )
            else:
                assert ValueError, f'Error model type: {self.model_type}'
        return sample


class LoadDataForImputation(LoadDataset):
    """Load all data for imputation, we don't need do any artificial mask here,
    just input original data into models and let them impute missing values"""

    def __init__(self, file_path, set_name, seq_len, feature_num, model_type):
        super(LoadDataForImputation, self).__init__(file_path, seq_len, feature_num, model_type)
        with h5py.File(self.file_path, 'r') as hf:  # read data from h5 file
            self.X = hf[set_name]['X'][:]
        self.missing_mask = (~np.isnan(self.X)).astype(np.float32)
        self.X = np.nan_to_num(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.model_type in ['Transformer', 'SAITS']:
            sample = (
                torch.tensor(idx),
                torch.from_numpy(self.X[idx].astype('float32')),
                torch.from_numpy(self.missing_mask[idx].astype('float32'))
            )
        elif self.model_type in ['BRITS', 'MRNN']:
            forward = {'X': self.X[idx], 'missing_mask': self.missing_mask[idx],
                       'deltas': parse_delta(self.missing_mask[idx], self.seq_len, self.feature_num)}

            backward = {'X': np.flip(forward['X'], axis=0).copy(),
                        'missing_mask': np.flip(forward['missing_mask'], axis=0).copy()}
            backward['deltas'] = parse_delta(backward['missing_mask'], self.seq_len, self.feature_num)
            sample = (
                torch.tensor(idx),
                # for forward
                torch.from_numpy(forward['X'].astype('float32')),
                torch.from_numpy(forward['missing_mask'].astype('float32')),
                torch.from_numpy(forward['deltas'].astype('float32')),
                # for backward
                torch.from_numpy(backward['X'].astype('float32')),
                torch.from_numpy(backward['missing_mask'].astype('float32')),
                torch.from_numpy(backward['deltas'].astype('float32')),
            )
        else:
            assert ValueError, f'Error model type: {self.model_type}'
        return sample


class UnifiedDataLoader:
    def __init__(self, dataset_path, seq_len, feature_num, model_type, batch_size=1024, num_workers=4,
                 masked_imputation_task=False):
        """
        dataset_path: path of directory storing h5 dataset;
        seq_len: sequence length, i.e. time steps;
        feature_num: num of features, i.e. feature dimensionality;
        batch_size: size of mini batch;
        num_workers: num of subprocesses for data loading;
        model_type: model type, determine returned values;
        masked_imputation_task: whether to return data for masked imputation task, only for training/validation sets;
        """
        self.dataset_path = os.path.join(dataset_path, 'datasets.h5')
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.masked_imputation_task = masked_imputation_task

        self.train_dataset, self.train_loader, self.train_set_size = None, None, None
        self.val_dataset, self.val_loader, self.val_set_size = None, None, None
        self.test_dataset, self.test_loader, self.test_set_size = None, None, None

    def get_train_val_dataloader(self):
        self.train_dataset = LoadTrainDataset(self.dataset_path, self.seq_len, self.feature_num, self.model_type,
                                              self.masked_imputation_task)
        self.val_dataset = LoadValTestDataset(self.dataset_path, 'val', self.seq_len, self.feature_num, self.model_type)
        self.train_set_size = self.train_dataset.__len__()
        self.val_set_size = self.val_dataset.__len__()
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.val_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return self.train_loader, self.val_loader

    def get_test_dataloader(self):
        self.test_dataset = LoadValTestDataset(self.dataset_path, 'test', self.seq_len, self.feature_num,
                                               self.model_type)
        self.test_set_size = self.test_dataset.__len__()
        self.test_loader = DataLoader(self.test_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return self.test_loader

    def prepare_dataloader_for_imputation(self, set_name):
        data_for_imputation = LoadDataForImputation(self.dataset_path, set_name, self.seq_len, self.feature_num,
                                                    self.model_type)
        dataloader_for_imputation = DataLoader(data_for_imputation, self.batch_size, shuffle=False)
        return dataloader_for_imputation

    def prepare_all_data_for_imputation(self):
        train_set_for_imputation = self.prepare_dataloader_for_imputation('train')
        val_set_for_imputation = self.prepare_dataloader_for_imputation('val')
        test_set_for_imputation = self.prepare_dataloader_for_imputation('test')
        return train_set_for_imputation, val_set_for_imputation, test_set_for_imputation
