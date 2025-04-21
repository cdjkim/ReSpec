import os
import json
import numpy as np
import pandas as pd

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
from pytorchvideo.transforms import ShortSideScale

from .base import BaseDataset
import pickle as pkl
from tqdm import tqdm

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275,  0.40821073)
OPENAI_DATASET_STD  = (0.26862954, 0.26130258, 0.27577711)

class LSMDC(BaseDataset):
    name = 'lsmdc'
    def __init__(self, args, transform=None):
        super(LSMDC, self).__init__()
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
        self.data_path = args.lsmdc_video_root
        ann_json_file  = f'{self.data_path}/lsmdc_{args.split}.pkl'
        iter2video_pairs_df_path = os.path.join(self.data_path, f"iter2video_pairs_{args.split}.csv")

        if os.path.exists(iter2video_pairs_df_path):
            iter2video_pairs_df = pd.read_csv(iter2video_pairs_df_path)
            print(f"Load iter2video_pairs_df from {iter2video_pairs_df_path} : {iter2video_pairs_df.shape}")

        else:
            iter2video_pairs_df = pd.DataFrame()
            with open(ann_json_file, 'rb') as f:
                self.ann_json = pkl.load(f)

            for video_path in tqdm(list(self.ann_json.keys())):
                video_id = video_path.split('/')[-1]
                video_key = video_path
                video_path = video_path.split('raw_videos/')[-1]
                video_path = os.path.join(self.data_path, video_path)
                temp_df = pd.DataFrame({
                    'video_id'  : [video_id],
                    'video_path': [video_path],
                    'text'      : [self.ann_json[video_key]],
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
        video_path = data['video_path']
        raw_text   = data['text']
        start      = None
        end        = None

        if 'Unnamed: 0' in data:
            unique_id = data['Unnamed: 0']
        else:
            unique_id = data.name
        ####################################
        # custom here
        ####################################
        frames, flag = self._get_frames(video_id=video_id,
                                        video_path=video_path,
                                        start=start, end=end)
        return unique_id, frames, raw_text, flag
