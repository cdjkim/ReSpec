# usage example
# python auto_evaluation.py -i cc3m_btadapter -u sangwoo.moon --result_only

import os
import subprocess
import argparse
from IPython import embed_kernel
from IPython.terminal.embed import embed
import yaml
import random
import getpass
import logging
import multiprocessing
import socket
import time

from glob import glob
import logging

import numpy as np
import pandas as pd
from mmengine.config import Config

from tools.test import main as test_main

def remove_prefix(word):
    word = word.lstrip('retrieval/')
    return word

def main(args):
    from pathlib import Path
    unique_id      = args.unique_id
    save_dir       = args.pretrained_dir
    pretrained_dir = save_dir
    masterport_min = 49152
    masterport_max = 65535

    # checkpoint setup
    if args.final_epoch:
        checkpoint_file = glob(f"{pretrained_dir}/epoch_*.pth")
        print('*******************')
        print(f"{pretrained_dir}/epoch_*.pth")
        print(checkpoint_file)
        print('*******************')
        checkpoint_file.sort()
        checkpoint_file = checkpoint_file[-1]
    else:
        checkpoint_file = glob(f"{pretrained_dir}/epoch_1.pth")[0]
    print(f"[TEST] Load checkpoint from {checkpoint_file}")

    # task setup
    if args.task == 'all':
        task_list = ['msrvtt', 'didemo', 'activitynet', 'youcook', 'lsmdc']
    else:
        NotImplementedError

    print(f"[TEST] Load checkpoint from {checkpoint_file}")
    for task in task_list:
        print(f'[[{task.upper()}_TEST START]]')
        config_file = f"configs/btadapter/zero-shot/{task}_btadapterl14.py"
        work_dir    = f"{save_dir}/{task}"
        flag_path   = os.path.join(work_dir, 'test_done')

        if not os.path.exists(flag_path):
            random_port = random.randint(masterport_min, masterport_max)
            print(f'evaluating {task} at port {random_port}')

            # specific args
            args.config      = config_file
            args.checkpoint  = checkpoint_file
            args.work_dir    = work_dir

            # default args
            args.dump       = None
            args.show       = False
            args.show_dir   = None
            args.interval   = 1
            args.wait_time  = 2
            args.launcher   = "pytorch"
            args.local_rank = 0

            test_main(args)
            Path(flag_path).touch()
            print(f'[[{task.upper()}_TEST DONE FLAG GENERATED]]')
        else:
            print(f'[[{task.upper()}_TEST DONE FLAG CHECKED]]')

    final_perf_lst = []
    result_lst = []
    for task in task_list:
        result_file_list = glob(f"{save_dir}/{task}/*/*.log")
        result_file_list.sort()
        result_file = result_file_list[-1]

        with open(result_file) as f:
            for line in f: pass
            last_index = 11 if task == 'vttqa' else 19
            result = line.split()[9:last_index]
            result = list(map(remove_prefix, result))
            result = ' '.join(result)
            result_lst.append(result)
            print(f'[{unique_id}][{task}]: {result}')
            final_perf_lst.append(f'[{unique_id}][{task}]: {result}\n')

    # read data len
    data_count_log = f'{save_dir}/data_count.log'
    with open(data_count_log, 'rb') as file:
        data_len = file.readline()

    # record final performance
    final_perf_log = f'{save_dir}/final_result.txt'
    try:
        final_perf_txt = f"{unique_id},{int(data_len)},,,,,"
    except Exception as e:
        final_perf_txt = f"{unique_id},-1,,,,,"

    print(unique_id,',',data_len,',,,,,',end='') # ignore duration/ratio
    for perf in result_lst:
        final_perf_txt = final_perf_txt + perf.strip() + ','
        print(perf.strip(), ',', end='')

    with open(final_perf_log, 'w') as file:
        file.write(final_perf_txt)
