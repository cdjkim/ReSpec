#!/usr/bin/env python3
from argparse import ArgumentParser
from pprint import pprint
import os
import resource
import random
import yaml
import torch
import colorful
import numpy as np
from tensorboardX import SummaryWriter
from data import DataScheduler
from models import MODEL


# Increase maximum number of open files.
# as suggested in https://github.com/pytorch/pytorch/issues/973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (24000, rlimit[1]))

parser = ArgumentParser()
parser.add_argument(    '--random_seed', '-r', type=int, default=0)
parser.add_argument(
    '--config', '-c', default='configs/msrvtt.yaml'
)
parser.add_argument('--log-dir', '-l')
parser.add_argument('--resume-ckpt', default=None)
parser.add_argument('--override', default='')
parser.add_argument('--command-file', '-f', default=None)


def main():
    args = parser.parse_args()

    # Use below for slurm setting.
    # slurm_job_id = os.getenv('SLURM_JOB_ID', 'nojobid')
    # slurm_proc_id = os.getenv('SLURM_PROC_ID', None)

    # unique_identifier = str(slurm_job_id)
    # if slurm_proc_id is not None:
    #     unique_identifier = unique_identifier + "_" + str(slurm_proc_id)
    unique_identifier = ''

    # Load config
    config_path = args.config

    if args.resume_ckpt and not args.config:
        base_dir = os.path.dirname(os.path.dirname(args.resume_ckpt))
        config_path = os.path.join(base_dir, 'config.yaml')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    config = override_config(config, args.override)

    config['log_dir'] = os.path.join(args.log_dir, unique_identifier)
    config['command_file'] = args.command_file

    # print the configuration
    print(colorful.bold_white("configuration:").styled_string)
    pprint(config)
    print(colorful.bold_white("configuration end").styled_string)

    if args.resume_ckpt and not args.log_dir:
        config['log_dir'] = os.path.dirname(
            os.path.dirname(args.resume_ckpt)
        )

    # set seed
    if args.random_seed != 0:
        config['random_seed'] = args.random_seed
    else:
        config['random_seed'] = random.randint(0, 1000)

    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    if not args.resume_ckpt or args.config:
        config_save_path = os.path.join(config['log_dir'], 'config.yaml')
        yaml.dump(config, open(config_save_path, 'w'))
        print(colorful.bold_yellow('config saved to {}'.format(config['log_dir'])).styled_string)

    writer = SummaryWriter(config['log_dir'])

    # general model exp flow
    data_scheduler = DataScheduler(config, writer)
    model = MODEL[config['model_name']](config, data_scheduler, writer)

    model.filter()

    print(colorful.bold_pink("\nThank you and Good Job Computer").styled_string)

def override_config(config, override):
    # Override options
    for option in override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                raise ValueError('{} is not defined in config file. '
                                 'Failed to override.'.format(address))
            here = here[key]
        if keys[-1] not in here:
            raise ValueError('{} is not defined in config file. '
                             'Failed to override.'.format(address))
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':
    main()