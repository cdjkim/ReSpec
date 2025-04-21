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

class DiDeMo(BaseDataset):
    name = 'didemo'
    def __init__(self, args, transform=None):
        super(DiDeMo, self).__init__()
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

        # reference: https://github.com/whwu95/Cap4Video/blob/main/dataloaders/dataloader_didemo_retrieval.py
        self.data_path  = args.didemo_train_file
        self.video_root = args.didemo_video_root
        iter2video_pairs_df_path = os.path.join(self.data_path, f"iter2video_pairs_{args.split}.csv")
        video_id_path_dict   = os.path.join(self.data_path, f"{args.split}_list.txt")
        video_json_path_dict = os.path.join(self.data_path, f"{args.split}_data.json")

        caption_dict = {} # video_id : caption
        video_dict = {}   # video_id : video_path

        with open(video_id_path_dict, 'r') as fp:
            video_ids = [itm.strip().split('.')[0] for itm in fp.readlines()]
        with open(video_json_path_dict, 'r') as f:
            json_data = json.load(f)

        # Get caption dict
        for itm in json_data:
            description = itm["description"]
            times = itm["times"]
            video = itm["video"].split('.')[0]
            if video not in video_ids:
                continue

            start_ = np.mean([t_[0] for t_ in times]) * 5
            end_ = (np.mean([t_[1] for t_ in times]) + 1) * 5
            if video in caption_dict:
                caption_dict[video]["start"].append(start_)
                caption_dict[video]["end"].append(end_)
                caption_dict[video]["text"].append(description)
            else:
                caption_dict[video] = {}
                caption_dict[video]["start"] = [start_]
                caption_dict[video]["end"] = [end_]
                caption_dict[video]["text"] = [description]

        # Get video dict
        for root, dub_dir, video_files in os.walk(self.video_root):
            for video_file in video_files:
                video_id_ = video_file.split('.')[0]
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_

        video_ids = list(set(video_ids) & set(caption_dict.keys()) & set(video_dict.keys()))

        if os.path.exists(iter2video_pairs_df_path):
            iter2video_pairs_df = pd.read_csv(iter2video_pairs_df_path, index_col=0)
            print(f"Load iter2video_pairs_df from {iter2video_pairs_df_path} : {iter2video_pairs_df.shape}")
        else:
            iter2video_pairs_df = pd.DataFrame()
            for video_id in caption_dict.keys():
                if video_id not in video_ids:
                    continue
                caption = caption_dict[video_id]
                n_caption = len(caption['text'])
                for sub_id in range(n_caption):
                    temp_df = pd.DataFrame({
                        'video_id': [video_id],
                        'sub_id':   [sub_id],
                        'text':     [caption['text'][sub_id]],
                        'start':    [caption['start'][sub_id]],
                        'end':      [caption['end'][sub_id]],
                        'clean_mask': [True]
                        })
                    iter2video_pairs_df = pd.concat([iter2video_pairs_df, temp_df], ignore_index=True)
            iter2video_pairs_df.to_csv(iter2video_pairs_df_path)

        self.caption_dict = caption_dict
        self.video_dict   = video_dict
        self.ann_file     = iter2video_pairs_df

    def __len__(self):
        return len(self.ann_file)

    def __getitem__(self, idx):
        ####################################
        # custom here
        ####################################
        data       = self.ann_file.iloc[idx]
        unique_id  = idx
        video_id   = data['video_id']
        video_path = self.video_dict[video_id]
        raw_text   = data['text']
        start      = data['start']
        end        = data['end']
        ####################################
        # custom here
        ####################################
        frames, flag = self._get_frames(video_id=video_id,
                                        video_path=video_path,
                                        start=start, end=end)
        return unique_id, frames, raw_text, flag