import argparse
import math
import os
import warnings
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime

import h5py
import nni
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from Global_Config import RANDOM_SEED
from modeling.SA_models import SAITS, TransformerEncoder
from modeling.brits import BRITS
from modeling.mrnn import MRNN
from modeling.unified_dataloader import UnifiedDataLoader
from modeling.utils import Controller, setup_logger, save_model, load_model, check_saving_dir_for_model, \
    masked_mae_cal, masked_rmse_cal, masked_mre_cal

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
warnings.filterwarnings('ignore')  # if to ignore warnings

MODEL_DICT = {
    # Self-Attention (SA) based
    'Transformer': TransformerEncoder, 'SAITS': SAITS,
    # RNN based
    'BRITS': BRITS, 'MRNN': MRNN,
}
OPTIMIZER = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}


def read_arguments(arg_parser, cfg_parser):
    # file path
    arg_parser.dataset_base_dir = cfg_parser.get('file_path', 'dataset_base_dir')
    arg_parser.result_saving_base_dir = cfg_parser.get('file_path', 'result_saving_base_dir')
    # dataset info
    arg_parser.seq_len = cfg_parser.getint('dataset', 'seq_len')
    arg_parser.batch_size = cfg_parser.getint('dataset', 'batch_size')
    arg_parser.num_workers = cfg_parser.getint('dataset', 'num_workers')
    arg_parser.feature_num = cfg_parser.getint('dataset', 'feature_num')
    arg_parser.dataset_name = cfg_parser.get('dataset', 'dataset_name')
    arg_parser.dataset_path = os.path.join(arg_parser.dataset_base_dir, arg_parser.dataset_name)
    arg_parser.eval_every_n_steps = cfg_parser.getint('dataset', 'eval_every_n_steps')
    # training settings
    arg_parser.MIT = cfg_parser.getboolean('training', 'MIT')
    arg_parser.ORT = cfg_parser.getboolean('training', 'ORT')
    arg_parser.lr = cfg_parser.getfloat('training', 'lr')
    arg_parser.optimizer_type = cfg_parser.get('training', 'optimizer_type')
    arg_parser.weight_decay = cfg_parser.getfloat('training', 'weight_decay')
    arg_parser.device = cfg_parser.get('training', 'device')
    arg_parser.epochs = cfg_parser.getint('training', 'epochs')
    arg_parser.early_stop_patience = cfg_parser.getint('training', 'early_stop_patience')
    arg_parser.model_saving_strategy = cfg_parser.get('training', 'model_saving_strategy')
    arg_parser.max_norm = cfg_parser.getfloat('training', 'max_norm')
    arg_parser.imputation_loss_weight = cfg_parser.getfloat('training', 'imputation_loss_weight')
    arg_parser.reconstruction_loss_weight = cfg_parser.getfloat('training', 'reconstruction_loss_weight')
    # model settings
    arg_parser.model_name = cfg_parser.get('model', 'model_name')
    arg_parser.model_type = cfg_parser.get('model', 'model_type')
    return arg_parser


def summary_write_into_tb(summary_writer, info_dict, step, stage):
    """write summary into tensorboard file"""
    summary_writer.add_scalar(f'total_loss/{stage}', info_dict['total_loss'], step)
    summary_writer.add_scalar(f'imputation_loss/{stage}', info_dict['imputation_loss'], step)
    summary_writer.add_scalar(f'imputation_MAE/{stage}', info_dict['imputation_MAE'], step)
    summary_writer.add_scalar(f'reconstruction_loss/{stage}', info_dict['reconstruction_loss'], step)
    summary_writer.add_scalar(f'reconstruction_MAE/{stage}', info_dict['reconstruction_MAE'], step)


def result_processing(results):
    """process results and losses for each training step"""
    results['total_loss'] = torch.tensor(0.0, device=args.device)
    if args.model_type == 'BRITS':
        results['total_loss'] = results['consistency_loss'] * args.consistency_loss_weight
    results['reconstruction_loss'] = results['reconstruction_loss'] * args.reconstruction_loss_weight
    results['imputation_loss'] = results['imputation_loss'] * args.imputation_loss_weight
    if args.MIT:
        results['total_loss'] += results['imputation_loss']
    if args.ORT:
        results['total_loss'] += results['reconstruction_loss']
    return results


def process_each_training_step(results, optimizer, val_dataloader, training_controller, summary_writer, logger):
    """process each training step and return whether to early stop"""
    state_dict = training_controller(stage='train')
    # apply gradient clipping if args.max_norm != 0
    if args.max_norm != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
    results['total_loss'].backward()
    optimizer.step()

    summary_write_into_tb(summary_writer, results, state_dict['train_step'], 'train')
    if state_dict['train_step'] % args.eval_every_n_steps == 0:
        state_dict_from_val = validate(model, val_dataloader, summary_writer, training_controller, logger)
        if state_dict_from_val['should_stop']:
            logger.info(f'Early stopping worked, stop now...')
            return True
    return False


def model_processing(data, model, stage,
                     # following arguments are only required in the training stage
                     optimizer=None, val_dataloader=None, summary_writer=None, training_controller=None, logger=None):
    if stage == 'train':
        optimizer.zero_grad()
        if not args.MIT:
            if args.model_type in ['BRITS', 'MRNN']:
                indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas = \
                    map(lambda x: x.to(args.device), data)
                inputs = {'indices': indices, 'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                          'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
            else:  # then for self-attention based models, i.e. Transformer/SAITS
                indices, X, missing_mask = map(lambda x: x.to(args.device), data)
                inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask}
            results = result_processing(model(inputs, stage))
            early_stopping = process_each_training_step(results, optimizer, val_dataloader, training_controller,
                                                        summary_writer, logger)
        else:
            if args.model_type in ['BRITS', 'MRNN']:
                indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas, X_holdout, \
                indicating_mask = map(lambda x: x.to(args.device), data)
                inputs = {'indices': indices, 'X_holdout': X_holdout, 'indicating_mask': indicating_mask,
                          'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                          'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
            else:
                indices, X, missing_mask, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)
                inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask,
                          'X_holdout': X_holdout, 'indicating_mask': indicating_mask}
            results = result_processing(model(inputs, stage))
            early_stopping = process_each_training_step(results, optimizer, val_dataloader,
                                                        training_controller, summary_writer, logger)
        return early_stopping

    else:  # in val/test stage
        if args.model_type in ['BRITS', 'MRNN']:
            indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas, X_holdout, \
            indicating_mask = map(lambda x: x.to(args.device), data)
            inputs = {'indices': indices, 'X_holdout': X_holdout, 'indicating_mask': indicating_mask,
                      'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                      'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
            inputs['missing_mask'] = inputs['forward']['missing_mask']  # for error calculation in validation stage
        else:
            indices, X, missing_mask, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)
            inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask,
                      'X_holdout': X_holdout, 'indicating_mask': indicating_mask}
        results = model(inputs, stage)
        results = result_processing(results)
        return inputs, results


def train(model, optimizer, train_dataloader, test_dataloader, summary_writer, training_controller, logger):
    for epoch in range(args.epochs):
        early_stopping = False
        args.final_epoch = True if epoch == args.epochs - 1 else False
        for idx, data in enumerate(train_dataloader):
            model.train()
            early_stopping = model_processing(data, model, 'train', optimizer, test_dataloader, summary_writer,
                                              training_controller, logger)
            if early_stopping:
                break
        if early_stopping:
            break
        training_controller.epoch_num_plus_1()
    logger.info('Finished all epochs. Stop training now.')


def validate(model, val_iter, summary_writer, training_controller, logger):
    model.eval()
    evalX_collector, evalMask_collector, imputations_collector = [], [], []
    total_loss_collector, imputation_loss_collector, reconstruction_loss_collector, reconstruction_MAE_collector = [], [], [], []

    with torch.no_grad():
        for idx, data in enumerate(val_iter):
            inputs, results = model_processing(data, model, 'val')
            evalX_collector.append(inputs['X_holdout'])
            evalMask_collector.append(inputs['indicating_mask'])
            imputations_collector.append(results['imputed_data'])

            total_loss_collector.append(results['total_loss'].data.cpu().numpy())
            reconstruction_MAE_collector.append(results['reconstruction_MAE'].data.cpu().numpy())
            reconstruction_loss_collector.append(results['reconstruction_loss'].data.cpu().numpy())
            imputation_loss_collector.append(results['imputation_loss'].data.cpu().numpy())

        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        imputation_MAE = masked_mae_cal(imputations_collector, evalX_collector, evalMask_collector)
    info_dict = {'total_loss': np.asarray(total_loss_collector).mean(),
                 'reconstruction_loss': np.asarray(reconstruction_loss_collector).mean(),
                 'imputation_loss': np.asarray(imputation_loss_collector).mean(),
                 'reconstruction_MAE': np.asarray(reconstruction_MAE_collector).mean(),
                 'imputation_MAE': imputation_MAE.cpu().numpy().mean()}
    state_dict = training_controller('val', info_dict, logger)
    summary_write_into_tb(summary_writer, info_dict, state_dict['val_step'], 'val')
    if args.param_searching_mode:
        nni.report_intermediate_result(info_dict['imputation_MAE'])
        if args.final_epoch or state_dict['should_stop']:
            nni.report_final_result(state_dict['best_imputation_MAE'])

    if (state_dict['save_model'] and args.model_saving_strategy) or args.model_saving_strategy == 'all':
        saving_path = os.path.join(
            args.model_saving, 'model_trainStep_{}_valStep_{}_imputationMAE_{:.4f}'.
                format(state_dict['train_step'], state_dict['val_step'], info_dict['imputation_MAE']))
        save_model(model, optimizer, state_dict, args, saving_path)
        logger.info(f'Saved model -> {saving_path}')
    return state_dict


def test_trained_model(model, test_dataloader):
    logger.info(f'Start evaluating on whole test set...')
    model.eval()
    evalX_collector, evalMask_collector, imputations_collector = [], [], []
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            inputs, results = model_processing(data, model, 'test')
            # collect X_holdout, indicating_mask and imputed data
            evalX_collector.append(inputs['X_holdout'])
            evalMask_collector.append(inputs['indicating_mask'])
            imputations_collector.append(results['imputed_data'])

        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        imputation_MAE = masked_mae_cal(imputations_collector, evalX_collector, evalMask_collector)
        imputation_RMSE = masked_rmse_cal(imputations_collector, evalX_collector, evalMask_collector)
        imputation_MRE = masked_mre_cal(imputations_collector, evalX_collector, evalMask_collector)

    assessment_metrics = {'imputation_MAE on the test set': imputation_MAE,
                          'imputation_RMSE on the test set': imputation_RMSE,
                          'imputation_MRE on the test set': imputation_MRE,
                          'trainable parameter num': args.total_params}
    with open(os.path.join(args.result_saving_path, 'overall_performance_metrics.out'), 'w') as f:
        logger.info('Overall performance metrics are listed as follows:')
        for k, v in assessment_metrics.items():
            logger.info(f'{k}: {v}')
            f.write(k + ':' + str(v))
            f.write('\n')


def impute_all_missing_data(model, train_data, val_data, test_data):
    logger.info(f'Start imputing all missing data in all train/val/test sets...')
    model.eval()
    imputed_data_dict = {}
    with torch.no_grad():
        for dataloader, set_name in zip([train_data, val_data, test_data], ['train', 'val', 'test']):
            indices_collector, imputations_collector = [], []
            for idx, data in enumerate(dataloader):
                if args.model_type in ['BRITS', 'MRNN']:
                    indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas = \
                        map(lambda x: x.to(args.device), data)
                    inputs = {'indices': indices, 'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                              'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
                else:  # then for self-attention based models, i.e. Transformer/SAITS
                    indices, X, missing_mask = map(lambda x: x.to(args.device), data)
                    inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask}
                imputed_data, _ = model.impute(inputs)
                indices_collector.append(indices)
                imputations_collector.append(imputed_data)

            indices_collector = torch.cat(indices_collector)
            indices = indices_collector.cpu().numpy().reshape(-1)
            imputations_collector = torch.cat(imputations_collector)
            imputations = imputations_collector.data.cpu().numpy()
            ordered = imputations[np.argsort(indices)]  # to ensure the order of samples
            imputed_data_dict[set_name] = ordered

    imputation_saving_path = os.path.join(args.result_saving_path, 'imputations.h5')
    with h5py.File(imputation_saving_path, 'w') as hf:
        hf.create_dataset('imputed_train_set', data=imputed_data_dict['train'])
        hf.create_dataset('imputed_val_set', data=imputed_data_dict['val'])
        hf.create_dataset('imputed_test_set', data=imputed_data_dict['test'])
    logger.info(f'Done saving all imputed data into {imputation_saving_path}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path of config file')
    parser.add_argument('--test_mode', dest='test_mode', action='store_true', help='test mode to test saved model')
    parser.add_argument('--param_searching_mode', dest='param_searching_mode', action='store_true',
                        help='use NNI to help search hyper parameters')
    args = parser.parse_args()
    assert os.path.exists(args.config_path), f'Given config file "{args.config_path}" does not exists'
    # load settings from config file
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config_path)
    args = read_arguments(args, cfg)

    if args.model_type in ['Transformer', 'SAITS']:  # if SA-based model
        args.input_with_mask = cfg.getboolean('model', 'input_with_mask')
        args.n_groups = cfg.getint('model', 'n_groups')
        args.n_group_inner_layers = cfg.getint('model', 'n_group_inner_layers')
        args.param_sharing_strategy = cfg.get('model', 'param_sharing_strategy')
        assert args.param_sharing_strategy in ['inner_group', 'between_group'], \
            'only "inner_group"/"between_group" sharing'
        args.d_model = cfg.getint('model', 'd_model')
        args.d_inner = cfg.getint('model', 'd_inner')
        args.n_head = cfg.getint('model', 'n_head')
        args.d_k = cfg.getint('model', 'd_k')
        args.d_v = cfg.getint('model', 'd_v')
        args.dropout = cfg.getfloat('model', 'dropout')
        args.diagonal_attention_mask = cfg.getboolean('model', 'diagonal_attention_mask')

        dict_args = vars(args)
        if args.param_searching_mode:
            tuner_params = nni.get_next_parameter()
            dict_args.update(tuner_params)
            experiment_id = nni.get_experiment_id()
            trial_id = nni.get_trial_id()
            args.model_name = f'{args.model_name}/{experiment_id}/{trial_id}'
            dict_args['d_k'] = dict_args['d_model'] // dict_args['n_head']
        model_args = {
            'device': args.device, 'MIT': args.MIT,
            # imputer args
            'n_groups': dict_args['n_groups'], 'n_group_inner_layers': args.n_group_inner_layers,
            'd_time': args.seq_len, 'd_feature': args.feature_num, 'dropout': dict_args['dropout'],
            'd_model': dict_args['d_model'], 'd_inner': dict_args['d_inner'], 'n_head': dict_args['n_head'],
            'd_k': dict_args['d_k'], 'd_v': dict_args['d_v'],
            'input_with_mask': args.input_with_mask,
            'diagonal_attention_mask': args.diagonal_attention_mask,
            'param_sharing_strategy': args.param_sharing_strategy,
        }
    elif args.model_type in ['BRITS', 'MRNN']:  # if RNN-based model
        if args.model_type == 'BRITS':
            args.consistency_loss_weight = cfg.getfloat('training', 'consistency_loss_weight')
        args.rnn_hidden_size = cfg.getint('model', 'rnn_hidden_size')

        dict_args = vars(args)
        if args.param_searching_mode:
            tuner_params = nni.get_next_parameter()
            dict_args.update(tuner_params)
            experiment_id = nni.get_experiment_id()
            trial_id = nni.get_trial_id()
            args.model_name = f'{args.model_name}/{experiment_id}/{trial_id}'
        model_args = {
            'device': args.device, 'MIT': args.MIT,
            # imputer args
            'seq_len': args.seq_len, 'feature_num': args.feature_num,
            'rnn_hidden_size': dict_args['rnn_hidden_size']
        }
    else:
        assert ValueError, f'Given model_type {args.model_type} is not in {MODEL_DICT.keys()}'

    # parameter insurance
    assert args.model_saving_strategy.lower() in ['all', 'best', 'none'], 'model saving strategy must be all/best/none'
    if args.model_saving_strategy.lower() == 'none':
        args.model_saving_strategy = False
    assert args.optimizer_type in OPTIMIZER.keys(), \
        f'optimizer type should be in {OPTIMIZER.keys()}, but get{args.optimizer_type}'
    assert args.device in ['cpu', 'cuda'], 'device should be cpu or cuda'

    time_now = datetime.now().__format__('%Y-%m-%d_T%H:%M:%S')
    args.model_saving, args.log_saving = check_saving_dir_for_model(args, time_now)
    logger = setup_logger(args.log_saving + '_' + time_now, 'w')
    logger.info(f'args: {args}')
    logger.info(f'Config file path: {args.config_path}')
    logger.info(f'Model name: {args.model_name}')

    unified_dataloader = UnifiedDataLoader(args.dataset_path, args.seq_len, args.feature_num, args.model_type,
                                           args.batch_size, args.num_workers, args.MIT)
    model = MODEL_DICT[args.model_type](**model_args)
    args.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Num of total trainable params is: {args.total_params}')

    # if utilize GPU and GPU available, then move
    if 'cuda' in args.device and torch.cuda.is_available():
        model = model.to(args.device)

    if args.test_mode:
        logger.info('Entering testing mode...')
        args.model_path = cfg.get('test', 'model_path')
        args.save_imputations = cfg.getboolean('test', 'save_imputations')
        args.result_saving_path = cfg.get('test', 'result_saving_path')
        os.makedirs(args.result_saving_path) if not os.path.exists(args.result_saving_path) else None
        model = load_model(model, args.model_path, logger)
        test_dataloader = unified_dataloader.get_test_dataloader()
        test_trained_model(model, test_dataloader)
        if args.save_imputations:
            train_data, val_data, test_data = unified_dataloader.prepare_all_data_for_imputation()
            impute_all_missing_data(model, train_data, val_data, test_data)
    else:  # in the training mode
        logger.info(f'Creating {args.optimizer_type} optimizer...')

        optimizer = OPTIMIZER[args.optimizer_type](model.parameters(), lr=dict_args['lr'],
                                                   weight_decay=args.weight_decay)
        logger.info('Entering training mode...')
        train_dataloader, val_dataloader = unified_dataloader.get_train_val_dataloader()
        training_controller = Controller(args.early_stop_patience)

        train_set_size = unified_dataloader.train_set_size
        logger.info(f'train set len is {train_set_size}, batch size is {args.batch_size},'
                    f'so each epoch has {math.ceil(train_set_size / args.batch_size)} steps')

        tb_summary_writer = SummaryWriter(os.path.join(args.log_saving, 'tensorboard_' + time_now))
        train(model, optimizer, train_dataloader, val_dataloader, tb_summary_writer, training_controller, logger)

    logger.info('All Done.')
