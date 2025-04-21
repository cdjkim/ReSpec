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

class YouCook(BaseDataset):
    os.environ['DECORD_EOF_RETRY_MAX'] = '20480'
    name = 'youcook'
    def __init__(self, args, transform=None):
        super(YouCook, self).__init__()
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
        # YouCook2 use validation for testing
        self.data_path  = args.youcook_train_file
        self.video_root = args.youcook_video_root
        iter2video_pairs_df_path = os.path.join(self.data_path, f"iter2video_pairs_{args.split}.csv")

        if args.split == 'train':
            ann_file_name = os.path.join(self.data_path, 'youcookii_annotations_trainval.json')
        elif args.split == 'test':
            ann_file_name = os.path.join(self.data_path, 'youcookii_annotations_trainval.json')
        else:
            print(f'{self.name} {args.split} data is not implemented')
            NotImplementedError

        with open(ann_file_name, 'r') as f:
            self.ann_json = json.load(f)['database']

        if os.path.exists(iter2video_pairs_df_path):
            iter2video_pairs_df = pd.read_csv(iter2video_pairs_df_path)
            print(f"Load iter2video_pairs_df from {iter2video_pairs_df_path} : {iter2video_pairs_df.shape}")
        else:
            iter2video_pairs_df = pd.DataFrame()
            for video_id in list(self.ann_json.keys()):
                subset = self.ann_json[video_id]['subset']
                recipe = self.ann_json[video_id]['recipe_type']

                if (args.split == 'train') and (subset == 'validation'):
                    continue
                if (args.split == 'test') and (subset == 'training'):
                    continue

                video_path = os.path.join(
                    self.video_root, subset, recipe, f'{video_id}.mp4')
                if not os.path.exists(video_path):
                    video_path = os.path.join(
                        self.video_root, subset, recipe, f'{video_id}.mkv')
                    if not os.path.exists(video_path):
                        video_path = os.path.join(
                            self.video_root, subset, recipe, f'{video_id}.webm')


                annotation = self.ann_json[video_id]['annotations']
                for sub_id in range(len(annotation)):
                    temp_df = pd.DataFrame({
                        'video_id': [video_id],
                        'video_path': [video_path],
                        'sub_id':   [sub_id],
                        'text':     [annotation[sub_id]['sentence']],
                        'start':    [annotation[sub_id]['segment'][0]],
                        'end':      [annotation[sub_id]['segment'][1]],
                        'clean_mask': [True]
                        })
                    iter2video_pairs_df = pd.concat([iter2video_pairs_df, temp_df], ignore_index=True)
            iter2video_pairs_df.to_csv(iter2video_pairs_df_path)

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
        start      = data['start']
        end        = data['end']

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
