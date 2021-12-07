"""
The simple RNN classification model for imputed dataset PhysioNet-2012.
"""
import argparse
import os
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Global_Config import RANDOM_SEED
from modeling.utils import cal_classification_metrics
from modeling.utils import setup_logger

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class LoadImputedDataAndLabel(Dataset):
    def __init__(self, imputed_data, labels):
        self.imputed_data = imputed_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.imputed_data[idx].astype('float32')),
            torch.from_numpy(self.labels[idx].astype('float32')),
        )


class ImputedDataLoader:
    def __init__(self, original_data_path, imputed_data_path, seq_len, feature_num, batch_size=128, num_workers=4):
        """
        original_data_path: path of original dataset, which contains classification labels
        imputed_data_path: path of imputed data
        """
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.batch_size = batch_size
        self.num_workers = num_workers

        with h5py.File(imputed_data_path, 'r') as hf:
            imputed_train_set = hf['imputed_train_set'][:]
            imputed_val_set = hf['imputed_val_set'][:]
            imputed_test_set = hf['imputed_test_set'][:]

        with h5py.File(original_data_path, 'r') as hf:
            train_set_labels = hf['train']['labels'][:]
            val_set_labels = hf['val']['labels'][:]
            test_set_labels = hf['test']['labels'][:]

        self.train_set = LoadImputedDataAndLabel(imputed_train_set, train_set_labels)
        self.val_set = LoadImputedDataAndLabel(imputed_val_set, val_set_labels)
        self.test_set = LoadImputedDataAndLabel(imputed_test_set, test_set_labels)

    def get_loaders(self):
        train_loader = DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(self.val_set, self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(self.test_set, self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader


class SimpleRNNClassification(torch.nn.Module):
    def __init__(self, feature_num, rnn_hidden_size, class_num):
        super().__init__()
        self.rnn = torch.nn.LSTM(feature_num, hidden_size=rnn_hidden_size, batch_first=True)
        self.fcn = torch.nn.Linear(rnn_hidden_size, class_num)

    def forward(self, data):
        hidden_states, _ = self.rnn(data)
        logits = self.fcn(hidden_states[:, -1, :])
        prediction_probabilities = torch.sigmoid(logits)
        return prediction_probabilities


def train(model, train_dataloader, val_dataloader, optimizer):
    patience = 20
    current_patience = patience
    best_ROCAUC = 0
    for epoch in range(args.epochs):
        model.train()
        for idx, data in enumerate(train_dataloader):
            X, y = map(lambda x: x.to(args.device), data)
            optimizer.zero_grad()
            probabilities = model(X)
            loss = F.binary_cross_entropy(probabilities, y)
            loss.backward()
            optimizer.step()

        # start val below
        model.eval()
        probability_collector, label_collector = [], []
        with torch.no_grad():
            for idx, data in enumerate(val_dataloader):
                X, y = map(lambda x: x.to(args.device), data)
                probabilities = model(X)
                probability_collector += probabilities.cpu().tolist()
                label_collector += y.cpu().tolist()
        probability_collector = np.asarray(probability_collector)
        label_collector = np.asarray(label_collector)
        classification_metrics = cal_classification_metrics(probability_collector, label_collector)
        if best_ROCAUC < classification_metrics['ROC_AUC']:
            current_patience = patience
            best_ROCAUC = classification_metrics['ROC_AUC']
            # save model
            saving_path = os.path.join(args.sub_model_saving,
                                       'model_epoch_{}_ROCAUC_{:.4f}'.format(epoch, best_ROCAUC))
            torch.save(model.state_dict(), saving_path)
        else:
            current_patience -= 1
        if current_patience == 0:
            break
    logger.info('All done. Training finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='model and log saving dir')
    parser.add_argument('--original_dataset_path', type=str, help='path of original dataset')
    parser.add_argument('--imputed_dataset_path', type=str, help='path of imputed dataset')
    parser.add_argument('--seq_len', type=int, help='sequence length')
    parser.add_argument('--feature_num', type=int, help='feature num')
    parser.add_argument('--rnn_hidden_size', type=int, help='RNN hidden size')
    parser.add_argument('--epochs', type=int, default=100, help='max training epochs')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--test_mode', dest='test_mode', action='store_true', help='test mode to test saved model')
    parser.add_argument('--saved_model_path', type=str, default=None, help='test mode to test saved model')
    parser.add_argument('--device', type=str, default='cuda', help='device to run model')

    args = parser.parse_args()
    if args.test_mode:
        assert args.saved_model_path is not None, 'saved_model_path must be provided in test mode'

    # create dirs
    time_now = datetime.now().__format__('%Y-%m-%d_T%H:%M:%S')
    log_saving = os.path.join(args.root_dir, 'logs')
    model_saving = os.path.join(args.root_dir, 'models')
    args.sub_model_saving = os.path.join(model_saving, time_now)
    [os.makedirs(dir_) for dir_ in [model_saving, log_saving, args.sub_model_saving] if not os.path.exists(dir_)]
    # create logger
    logger = setup_logger(os.path.join(log_saving, 'log_' + time_now), 'w')
    logger.info(f'args: {args}')
    # build models and dataloaders
    model = SimpleRNNClassification(args.feature_num, args.rnn_hidden_size, 1)
    dataloader = ImputedDataLoader(args.original_dataset_path, args.imputed_dataset_path,
                                   args.seq_len, args.feature_num, 128)
    train_set_loader, val_set_loader, test_set_loader = dataloader.get_loaders()
    if 'cuda' in args.device and torch.cuda.is_available():
        model = model.to(args.device)

    if not args.test_mode:
        logger.info('Start training...')
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        train(model, train_set_loader, val_set_loader, optimizer)
    else:
        logger.info('Start testing...')
        checkpoint = torch.load(args.saved_model_path)
        model.load_state_dict(checkpoint)
        model.eval()
        probability_collector, label_collector = [], []
        for idx, data in enumerate(test_set_loader):
            X, y = map(lambda x: x.to(args.device), data)
            probabilities = model(X)
            probability_collector += probabilities.cpu().tolist()
            label_collector += y.cpu().tolist()
        probability_collector = np.asarray(probability_collector)
        label_collector = np.asarray(label_collector)
        classification_metrics = cal_classification_metrics(probability_collector, label_collector)
        for k, v in classification_metrics.items():
            logger.info(f'{k}: {v}')
