# Copyright (c) OpenMMLab. All rights reserved.

import os
import argparse
import os.path as osp


from mmengine.config import Config, DictAction
from mmaction.engine.runner import myFlexibleRunner
from mmengine.runner import Runner

from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

from auto_evaluation import main as eval_main

size_based_auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy, min_num_params=1e7)

from pathlib import Path
import pandas as pd
import numpy as np
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train a action recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--load_from', default=None,)
    parser.add_argument(
        '--resume',

        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument(
        '--diff-rank-seed',
        action='store_true',
        help='whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
        '--use-fsdp',
        action='store_true')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if args.load_from is not None:
        cfg.load_from = args.load_from

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # set random seeds
    if cfg.get('randomness', None) is None:
        cfg.randomness = dict(
            seed=args.seed,
            diff_rank_seed=args.diff_rank_seed,
            deterministic=args.deterministic)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
        if 'train_ann_file' in args.cfg_options:
            cfg['train_dataloader']['dataset']['ann_file'] = args.cfg_options['train_ann_file']
        if 'train_split_file' in args.cfg_options:
            cfg['train_dataloader']['dataset']['split_file'] = args.cfg_options['split_file']
        if 'random_seed' in args.cfg_options:
            cfg['train_dataloader']['dataset']['random_seed'] = args.cfg_options['random_seed']
        if 'random_sample' in args.cfg_options:
            cfg['train_dataloader']['dataset']['random_sample'] = args.cfg_options['random_sample']
        if 'train_epoch' in args.cfg_options:
            cfg['train_cfg']['max_epochs']     = args.cfg_options['train_epoch']
            cfg['param_scheduler'][1]['T_max'] = args.cfg_options['train_epoch'] - 0.5
            cfg['param_scheduler'][1]['end']   = args.cfg_options['train_epoch']
        if 'online_training' in args.cfg_options:
            cfg['train_dataloader']['sampler']['shuffle'] = False if args.cfg_options['online_training'] else True
        if 'train_ann_file_cc3m' in args.cfg_options:
            print("SET (train_ann_file_cc3m, webvid2m) in cfg")
            cfg['train_dataloader']['dataset']['ann_file'] = (args.cfg_options['train_ann_file_cc3m'], args.cfg_options['train_ann_file_webvid2m'])

    return cfg


def main():
    # 指定 FSDPStrategy 并配置参数
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    # merge cli arguments to config
    if args.use_fsdp:
        strategy = dict(
            type='FSDPStrategy',
            model_wrapper=dict(auto_wrap_policy=size_based_auto_wrap_policy))
        cfg['strategy'] = strategy
        RUNNER = myFlexibleRunner
    else:
        RUNNER = Runner

    print('[[TRAIN START]]')
    data_count_path = os.path.join(cfg.work_dir, 'data_count.log')
    flag_path       = os.path.join(cfg.work_dir, 'train_done')
    if not os.path.exists(flag_path):
        # build the runner from config
        runner = RUNNER.from_cfg(cfg)
        runner.train()
        Path(flag_path).touch()
        print('[[TRAIN DONE FLAG GENERATED]]')
    else:
        print('[[TRAIN DONE FLAG CHECKED]]')

    with open(data_count_path, 'w') as f:
        if 'train_ann_file_cc3m' in args.cfg_options:
            if cfg.train_ann_file_cc3m.endswith('.npz'):
                raw_dataframe = np.load(cfg.train_ann_file_cc3m, allow_pickle=True)['data']
                dataset_cc3m = pd.DataFrame(raw_dataframe, columns = ['video_id', 'start', 'end', 'text'])
            elif cfg.train_ann_file_cc3m.endswith('.csv'):
                dataset_cc3m = pd.read_csv(cfg.train_ann_file_cc3m)
            elif cfg.train_ann_file_cc3m.endswith('.pkl'):
                dataset_cc3m = pd.read_pickle(cfg.train_ann_file_cc3m)

            if cfg.train_ann_file_webvid2m.endswith('.npz'):
                raw_dataframe = np.load(cfg.train_ann_file_webvid2m, allow_pickle=True)['data']
                dataset_webvid2m = pd.DataFrame(raw_dataframe, columns = ['video_id', 'start', 'end', 'text'])
            elif cfg.train_ann_file_webvid2m.endswith('.csv'):
                dataset_webvid2m = pd.read_csv(cfg.train_ann_file_webvid2m)
            elif cfg.train_ann_file_webvid2m.endswith('.pkl'):
                dataset_webvid2m = pd.read_pickle(cfg.train_ann_file_webvid2m)

            print(f"dataset_size: {len(dataset_cc3m) + len(dataset_webvid2m)}")
            print(len(dataset_cc3m) + len(dataset_webvid2m))
            args.dataset_size = len(dataset_cc3m) + len(dataset_webvid2m)
            f.write(f"{args.dataset_size}")

        else:
            if cfg.train_ann_file.endswith('.npz'):
                raw_dataframe = np.load(cfg.train_ann_file, allow_pickle=True)['data']
                dataset = pd.DataFrame(raw_dataframe, columns = ['video_id', 'start', 'end', 'text'])
            elif cfg.train_ann_file.endswith('.csv'):
                dataset = pd.read_csv(cfg.train_ann_file)
            elif cfg.train_ann_file.endswith('.pkl'):
                dataset = pd.read_pickle(cfg.train_ann_file)
            args.dataset_size = len(dataset)
            f.write(f"{args.dataset_size}")

        time.sleep(10)

        print('[[TEST START]]')
        # set default args
        # set default args (maybe useless)
        args.unique_id      = cfg.work_dir.split('/')[-1]
        args.pretrained_dir = cfg.work_dir
        args.task           = 'all'
        args.result_only    = False
        args.one_line       = True
        args.final_epoch    = True

        eval_main(args)

if __name__ == '__main__':
    main()
