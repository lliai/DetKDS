import random
import logging
import sys
import optuna
from train import main
from optuna.visualization import plot_param_importances
from mmcv import Config, DictAction
import argparse
import warnings
import os
from writing_config import over_write


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def get_best_value(opt):
    args = parse_args()
    over_write(opt['trans_func1'], opt['trans_func2'], opt['trans_func3'],
               opt['distance_func'], opt['gamma'], args.config)
    bbox_mAP = main(args)
    return bbox_mAP


def objective(trial):
    trans_func1 = trial.suggest_categorical('trans_func1',
                                            ['drop','no','satt','catt','mask'])
    trans_func2 = trial.suggest_categorical('trans_func2',
                                            ['no','scale_r1','scale_r2','multi_scale_r4','local_s1','local_s2','local_s4','scale'])
    trans_func3 = trial.suggest_categorical('trans_func3',
                                            ['no','bmm','mm','batch','channel','norm_HW','norm_C','norm_N','min_max_normalize','batchnorm'])
    distance_func = trial.suggest_categorical('distance_func',
                                              ["l1", "l2", "kl", "smooth_l1", "cos", "pear"])
    gamma = trial.suggest_categorical('gamma', [0.1, 0.5, 1, 5, 10, 20, 50])
    # kd_T = trial.suggest_categorical('kd_T', [1])
    # weight = trial.suggest_categorical('weight', ["no"])

    opt = dict()
    opt['trans_func1'] = trans_func1
    opt['trans_func2'] = trans_func2
    opt['trans_func3'] = trans_func3
    opt['distance_func'] = distance_func
    opt['gamma'] = gamma
    # opt['kd_T'] = kd_T
    # opt['weight'] = weight

    result = get_best_value(opt)

    return result


if __name__ == "__main__":
    study_name = "optuna-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(direction='maximize', sampler = optuna.samplers.NSGAIISampler(), 
                                storage=storage_name)
    study.optimize(objective, n_trials=20, n_jobs=1)
    plot_param_importances(study).show()
    optuna.visualization.plot_param_importances(study)

    # 展示的命令
    # pip install optuna-dashboard
    # optuna-dashboard sqlite:///optuna-study.db
