import os
import json
import numpy as np
import pandas as pd

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
from pytorchvideo.transforms import ShortSideScale

from .base import BaseDataset

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275,  0.40821073)
OPENAI_DATASET_STD  = (0.26862954, 0.26130258, 0.27577711)

class ActivityNet(BaseDataset):
    name = 'activitynet'
    def __init__(self, args, transform=None):
        super(ActivityNet, self).__init__()
        self.args       = args
        self.max_frames = args.max_frames
        self.max_words  = args.max_words
        if transform != None:
            self.transform = transform
        else:
            self.transform = Compose([
                        Lambda(lambda x: x / 255.0),
                        NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                        ShortSideScale(size=224),
                        CenterCropVideo(224),
                        ])

        ####################################
        # custom here
        ####################################
        self.data_path  = args.activitynet_train_file
        self.video_root = args.activitynet_video_root
        iter2video_pairs_df_path = os.path.join(self.data_path, f"iter2video_pairs_{args.split}.csv")

        if args.split == 'train':
            ann_file_name = os.path.join(self.data_path, 'anet_ret_train.json')
        else:
            print(f'{self.name} {args.split} data is not implemented')

        with open(ann_file_name, 'r') as f:
            self.ann_json = json.load(f)

        if os.path.exists(iter2video_pairs_df_path):
            iter2video_pairs_df = pd.read_csv(iter2video_pairs_df_path)
            print(f"Load iter2video_pairs_df from {iter2video_pairs_df_path} : {iter2video_pairs_df.shape}")
        else:
            iter2video_pairs_df = pd.DataFrame()
            for i in range(len(self.ann_json)):
                video_id   = self.ann_json[i]['video'].split('.')[0]
                video_path = self.ann_json[i]['video']
                caption    = self.ann_json[i]['caption']
                timestamp  = self.ann_json[i]['timestamp']
                for sub_id in range(len(caption)):
                    temp_df = pd.DataFrame({
                        'video_id': [video_id],
                        'sub_id':   [sub_id],
                        'text':     [caption[sub_id]],
                        'start':    [timestamp[sub_id][0]],
                        'end':      [timestamp[sub_id][1]],
                        'clean_mask': [True]
                        })
                    iter2video_pairs_df = pd.concat([iter2video_pairs_df, temp_df], ignore_index=True)
            iter2video_pairs_df.to_csv(iter2video_pairs_df_path)
            iter2video_pairs_df = pd.read_csv(iter2video_pairs_df_path) # To get unique id, 'Unnamed: 0'

        self.ann_file = iter2video_pairs_df
        ####################################
        # custom here
        ####################################

    def __len__(self):
        return len(self.ann_file)

    def __getitem__(self, idx):
        ####################################
        # custom here
        ####################################
        data       = self.ann_file.iloc[idx]
        video_id   = data['video_id']
        video_path = os.path.join(self.video_root, f'{video_id}.mp4')
        raw_text   = data['text']
        start      = data['start']
        end        = data['end']

        if 'Unnamed: 0' in data:
            unique_id = data['Unnamed: 0']
        ####################################
        # custom here
        ####################################
        frames, flag = self._get_frames(video_id=video_id,
                                        video_path=video_path,
                                        start=start, end=end)
        return unique_id, frames, raw_text, flag
