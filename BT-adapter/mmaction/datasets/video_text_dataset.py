# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import csv
import os.path as osp
import numpy as np
import pandas as pd
import pickle as pkl
from glob import glob

from typing import Callable, Dict, List, Optional, Union

from mmengine.fileio import exists, list_from_file

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset
from .transforms.text_transforms import tokenize
from .video_dataset import VideoDataset


@DATASETS.register_module()
class MsrvttDataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        with open(self.ann_file) as f:
            video_dict = json.load(f)
            for filename, texts in video_dict.items():
                filename = osp.join(self.data_prefix['video'], filename)
                video_text_pairs = []
                for text in texts:
                    data_item = dict(filename=filename, text=text)
                    video_text_pairs.append(data_item)
                data_list.extend(video_text_pairs)

        return data_list


@DATASETS.register_module()
class DidemoDataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        with open(self.ann_file) as f:
            video_list = json.load(f)
            for video_dict in video_list:
                # for filename, texts in video_dict.items():
                filename = osp.join(self.data_prefix['video'], video_dict['video'])
                text = " ".join(video_dict['desc'])
                data_item = dict(filename=filename, text=text)
                data_list.append(data_item)
        return data_list


@DATASETS.register_module()
class VttQADataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        csv_data = pd.read_csv(self.ann_file, sep="\t")
        for video_id, a1, a2, a3, a4, a5, answer in zip(
                csv_data["vid_key"].values,
                csv_data["a1"].values,
                csv_data["a2"].values,
                csv_data["a3"].values,
                csv_data["a4"].values,
                csv_data["a5"].values,
                csv_data["answer"].values):
            video_id = video_id.replace("msr", "video")
            filename = osp.join(self.data_prefix['video'], video_id + '.mp4')
            for text in [a1, a2, a3, a4, a5]:
                data_item = dict(filename=filename, text=text)
                data_list.append(data_item)

        return data_list


@DATASETS.register_module()
class How2QADataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        csv_data = pd.read_csv(self.ann_file)
        for video_id, ext, start, end, question, a0, a1, a2, a3, in zip(
                csv_data["video_id"].values,
                csv_data["ext"].values,
                csv_data["start"].values,
                csv_data["end"].values,
                csv_data["question"].values,
                csv_data["a0"].values,
                csv_data["a1"].values,
                csv_data["a2"].values,
                csv_data["a3"].values,
                ):
            video_id_rsplit = video_id.rsplit('_', 2)
            start = int(video_id_rsplit[1]) + start
            end = int(video_id_rsplit[1]) + end
            video_id = video_id_rsplit[0]
            filename = osp.join(self.data_prefix['video'], f'{video_id}_{start}_{end}.{ext}')
            for text in [a0, a1, a2, a3]:
                data_item = dict(filename=filename, text=f'{question} {text}')
                data_list.append(data_item)

        return data_list


@DATASETS.register_module()
class YoucookDataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        if '_trainval.json' in self.ann_file:
            with open(self.ann_file) as f:
                trainval_data = json.load(f)['database']
                for video_id in trainval_data:
                    if trainval_data[video_id]['subset'] == 'validation':
                        continue
                    else:
                        recipe_type = trainval_data[video_id]['recipe_type']
                        filename = glob(osp.join(self.data_root, 'training', recipe_type, f"{video_id}.*"))[-1]
                        for ann in trainval_data[video_id]['annotations']:
                            start, end = ann['segment']
                            text = ann['sentence']
                            data_item = dict(filename=filename, start_sec=start, 
                                            end_sec = end, text=text)
                            data_list.append(data_item)
        else:
            with open(self.ann_file) as f:
                val_data = json.load(f)
                for video_data in val_data:
                    filename = osp.join(self.data_root, video_data['filename'])
                    data_item = dict(filename=filename, text=video_data['text'])
                    data_list.append(data_item)

        return data_list


@DATASETS.register_module()
class HowToAlignDataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        with open(self.ann_file) as f:
            data = json.load(f)
            for video_id, video_data in list(data.items())[:2]:
                num_clips = len(video_data['clip_start'])
                num_texts = len(video_data['text_str'])
                for i in range(max(num_clips, num_texts)):
                    idx = min(i, num_clips-1)
                    start = video_data['clip_start'][idx]
                    end = video_data['clip_end'][idx]

                    text = video_data['text_str'][i] if i < num_texts else ""
                    video_name = f"{video_id}_{start}_{end}.mp4" if i < num_clips else "placeholder.mp4"

                    filename = osp.join(self.data_root, video_name)
                    data_item = dict(filename=filename, text=text)
                    data_list.append(data_item)

        return data_list


@DATASETS.register_module()
class ActivityNetVideoDataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        with open(self.ann_file) as f:
            videos = f.readlines()
            for video in videos:
                video = video.strip()
                video,frame = video.split(',')
                frame_dir = osp.join(self.data_root,video)
                data_item = dict(file_name=video, frame_dir=frame_dir, total_frames=int(frame), filename_tmpl="{:0>6}.jpg", offset=1)
                data_list.append(data_item)

        return data_list


@DATASETS.register_module()
class ActivityNetRetDataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""
    def __init__(self, num_file=None, **kwargs) -> None:
        self.num_file = num_file
        super().__init__(**kwargs)

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        if "frame" in self.data_root :
            with open(self.num_file,'r') as f1:
                frame_number = json.load(f1)
            with open(self.ann_file,'r') as f2:
                videos = json.load(f2)
                for ann in videos:
                    video = ann['video'][:-4]
                    video = video.strip()
                    frame = frame_number[video]
                    frame_dir = osp.join(self.data_root,video)
                    caption = ' '.join(ann['caption'])
                    data_item = dict(filename=video, frame_dir=frame_dir, total_frames=int(frame),\
                                    filename_tmpl="img_{:0>5}.jpg", offset=0, text=caption)
                    data_list.append(data_item)
        else:
            with open(self.ann_file,'r') as f2:
                videos = json.load(f2)
                for ann in videos:
                    video_path = ann['video'][:-4]
                    video_path = osp.join(self.data_root, video_path)
                    for idx in range(len(ann['caption'])):
                        caption = ann['caption'][idx]
                        start   = ann['timestamp'][idx][0]
                        end     = ann['timestamp'][idx][1]
                        data_item = dict(filename=video_path, start_sec=start, end_sec=end, text=caption)
                        data_list.append(data_item)
        
        return data_list

@DATASETS.register_module()
class CC3MDataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    # csv format: │url│start│end│caption│id│duration│download│video_path
    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        if self.ann_file[-4:] == '.csv': df = pd.read_csv(self.ann_file)
        else: df = pd.read_pickle(self.ann_file)

        # replace video_path to data_root
        df['video_path'] = df['video_path'].str.replace(
            '/gallery_millet/chris.kim/data/videocc3m/clips_final',
            self.data_root)

        # randomly sample subset of df. (with random_seed)
        if hasattr(self,'random_sample'):
            df = df.sample(
                n=self.random_sample,
                replace=False,
                random_state=self.random_seed,)

        for i in range(len(df)):
            file_path = df.iloc[i]['video_path']
            text      = df.iloc[i]['caption']

            info      = {
                'filename' : file_path,
                'text'     : text
                }

            data_list.append(info)

        return data_list


@DATASETS.register_module()
class WebvidDataset_source(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        df = pd.read_pickle(self.ann_file)
        data_list = []

        for i in range(len(df)):
            row = df.iloc[i]
            video_id, video_dir = row['videoid'], row['page_dir']
            filename = os.path.join(self.data_root,video_dir,str(video_id)+'.mp4')
            info = {'filename':filename, 'text':row['name'].strip()}
            data_list.append(info)

        return data_list

@DATASETS.register_module()
class LsmdcDataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        with open(self.ann_file, 'rb') as f:
            video_dict = pkl.load(f)
            for video_dir, text in video_dict.items():
                video_dir = video_dir.replace(video_dir.split('/raw_videos')[0], self.data_root)
                info = {'filename':video_dir, 'text':text}
                data_list.append(info)

        return data_list

@DATASETS.register_module()
class ZeroShotClfDataset(VideoDataset):
    def __init__(self, class_path, label_offset=0, **kwargs):
        self.label_offset = label_offset
        super().__init__(**kwargs)

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split(self.delimiter)
            if self.multi_class:
                assert self.num_classes is not None
                filename, label = line_split[0], line_split[1:]
                label = list(map(int, label))
            else:
                filename, label = line_split
                label = int(label) + self.label_offset
            if self.data_prefix['video'] is not None:
                filename = osp.join(self.data_prefix['video'], filename)
            data_list.append(dict(filename=filename, label=label, text=[0]))
        return data_list
