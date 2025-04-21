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

class MSRVTT(BaseDataset):
    name = 'msrvtt'
    def __init__(self, args, transform = None):
        super(MSRVTT, self).__init__()
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

        self.video_root = args.msrvtt_video_root
        ann_file_name   = args.msrvtt_train_file
        with open(ann_file_name, 'r') as f:
            self.ann_file = json.load(f)
        self.video_id_list = list(self.ann_file.keys())

    def __len__(self):
        return len(self.ann_file)

    def __getitem__(self, idx):
        ####################################
        # custom here 
        ####################################
        video_id = self.video_id_list[idx]
        video_path = os.path.join(self.video_root, f'{video_id}.mp4')
        if self.args.split == 'train':
            raw_text = self.ann_file[video_id][0] # multi-setence in MSRVTT
            start    = None
            end      = None
        else:
            print(f'{self.name} {self.args.split} data is not implemented')
            NotImplementedError

        ####################################
        # custom here 
        ####################################
        frames, flag = self._get_frames(video_id=video_id, 
                                        video_path=video_path, 
                                        start=start, end=end)
        return video_id, frames, raw_text, flag