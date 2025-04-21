import numpy as np
from abc import ABC

import torch
import decord
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class BaseDataset(Dataset, ABC):
    name = 'base'
    dataset_size = 0

    def __init__(self):
        super().__init__()
        self.max_frames = 8
        self.max_words  = 77

    def __len__(self):
        return self.dataset_size

    def collate_fn(self, batch):
        return default_collate(batch)

    def _get_frames(self, video_id=None, video_path=None, text_id=None, start=None, end=None):
        """ Get video information
        INPUT:
            video_id  : video_id
            video_path: video_path
            text_id   : text_id
        OUTPUT:
            video_outputs: (3, max_frame, 224, 224), torch.tensor
                - max_frame : max frame per clip
        """
        try:
            # similar with LB pipeline
            decord.bridge.set_bridge('torch')
            decord_vr     = VideoReader(video_path, ctx=cpu(0), num_threads=1) # some error on multi-threading...(in youcook video)
            duration      = len(decord_vr)
            if start == None:
                frame_id_list = np.linspace(0, duration-1, self.max_frames, dtype=int)
            else: # add this for start-end sampling
                fps           = decord_vr.get_avg_fps()
                start_frame   = int(start * fps)
                end_frame     = min(int(end * fps), duration-1)
                frame_id_list = np.linspace(start_frame, end_frame, self.max_frames, dtype=int)
            video_data    = decord_vr.get_batch(frame_id_list)
            video_data    = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
            video_outputs = self.transform(video_data)
            return video_outputs, True

        except Exception as e:
            print(f'video_id {video_id}, text_id {text_id} sample is corrupted, {e}')
            return torch.zeros((3, self.max_frames, 224, 224), dtype=torch.float), False # bad clip-captions
