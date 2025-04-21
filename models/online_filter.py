from abc import ABC, abstractmethod

import os
import sys
import torch
import pickle as pkl
import random
import pandas as pd

from torch.utils.data import DataLoader
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)



class OnlineFilter(torch.nn.Module):
    def __init__(self, config, scheduler, writer):
        super(OnlineFilter, self).__init__()
        self.config      = config
        self.writer      = writer
        self.scheduler   = scheduler

        self.device      = config['device'] if torch.cuda.is_available() else 'cpu'
        self.result = {}
        self.result_in_singles = {}
        self.result_in_singles = {
                                'video_id': [],
                                'start': [],
                                'end': [],
                                'text': []
                                 }

        if 'modality' in config:
            self.modality = config['modality'].split('-')


        self.threshold = self.config['threshold']

        # prepare datalaoder
        torch.manual_seed(self.config['random_seed'])
        dataset = self.scheduler.datasets[self.config['data_name']]
        self.data_size = len(dataset)

        self.dataloader = DataLoader(
            dataset,
            batch_size  = self.config['batch_size'],
            num_workers = 8,
            drop_last   = False,
            pin_memory  = False, # better when training on GPU.
            shuffle     = False,
            collate_fn  = self.custom_collate_fn
            )

    def custom_collate_fn(self, batch):
        data = zip(*batch)
        video_id, real_text, start, end, orig_start, orig_end, \
            real_text_sim, embedding = data
        return video_id, real_text, start, end, orig_start, \
            orig_end, real_text_sim, embedding


    @abstractmethod
    def filter(self):
        pass

    def store_result(self, vid_id:str, text:str, start, end, orig_start, orig_end):
        already_exists = False
        if vid_id not in self.result.keys():
            self.result[vid_id] = {
                'text' : [text],
                'start': [str(start)],
                'end'  : [str(end)]
            }
        else:
            # cc3m has unique vid_ids.
            if self.config['data_name'] == 'cc3m' or self.config['data_name'] in 'webvid':
                already_exists = True
                return already_exists

            self.result[vid_id]['text'].append(text)
            self.result[vid_id]['start'].append(str(start))
            self.result[vid_id]['end'].append(str(end))

        return already_exists

    def save_result(self):
        # orig_fn = os.path.basename(self.config['json_path'])
        if self.config['split_path']:
            orig_fn = os.path.basename(self.config['split_path']) # {i}
        else:
            orig_fn = os.path.basename(self.config['hdf5_path']) # {i}
        orig_fn, ext = os.path.splitext(orig_fn)
        save_path = os.path.join(self.config['log_dir'], 'clean_meta_' + orig_fn)
        # os.makedirs(save_dir, exist_ok=True)

        if self.config['split_path']:
            save_path = os.path.join(os.path.dirname(save_path), '..',
                                            os.path.basename(save_path))

        print(f'save result in {save_path}')
        with open(save_path + '.pkl', 'wb') as f:
            pkl.dump(self.result, f, pkl.HIGHEST_PROTOCOL)

        if 'cc3m' in self.config['data_name']:
            meta_df = self.scheduler.datasets[self.config['data_name']].meta_data
            filtered_df = meta_df[meta_df['unique_index'].isin(self.result.keys())]
            save_path = save_path + '.csv'
            filtered_df.to_csv(save_path, index=False)
        elif 'webvid' in self.config['data_name']:
            meta_df = self.scheduler.datasets[self.config['data_name']].meta_data
            filtered_df = meta_df[meta_df['unique_id'].isin(self.result.keys())]
            save_path = save_path + '.pkl'
            filtered_df.to_pickle(save_path)

        print(f"sample #: {len(filtered_df)}")

        return save_path

    def gen_downstream_cmd(self, save_path):
        base_path = './'
        save_path = os.path.join(base_path, save_path)

        master_port = random.randint(49152, 65535)
        data_name = self.config['data_name']
        if 'cc3m' in data_name:
            data_config_file = f'cc3m.py'
        elif 'webvid' in data_name:
            data_config_file = f'webvid2m.py'
        else:
            raise NotImplementedError(f"{data_name} is an unknown dataset")

        unique_dir = os.path.normpath(self.config['log_dir']).split(os.sep)[-2]
        down_cmd = (f"python -m torch.distributed.run --nproc_per_node=4 --master_port={master_port} "
        f"tools/train.py configs/btadapter/pretrain/base/{data_config_file} "
        f"--cfg-options work_dir=./work_dirs/{unique_dir} "
        f"train_ann_file={save_path} train_epoch=1 --no-validate; "
        f"python auto_evaluation.py --one_line --final_epoch -i={unique_dir}")

        print("RUN Downstream CMD: ")
        print(down_cmd)
        return [down_cmd]

