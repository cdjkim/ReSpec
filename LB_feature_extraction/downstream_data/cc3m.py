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

class CC3M(BaseDataset):
    name = 'cc3m'
    def __init__(self, args, transform=None):
        super(CC3M, self).__init__()
        self.args   = args
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
        self.video_root = args.cc3m_video_root
        ann_file_name = args.cc3m_train_file
        self.df = pd.read_cvs(ann_file_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data      = self.df.iloc[idx]
        unique_id = data['unique_index'] # use idx as unique identifier (name for dataset in h5 file)
        video_id  = data['id']
        video_path = os.path.join(self.video_root, f"{video_id}.mp4")
        raw_text  = data['caption']
        start     = None
        end       = None
        frames, valid_flag = self._get_frames(video_id=video_id, video_path=video_path, 
                                            text_id=None, start=start, end=end)

        return unique_id, frames, raw_text, valid_flag