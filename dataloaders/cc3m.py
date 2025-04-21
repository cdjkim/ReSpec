import os
import orjson as json

import numpy as np
import pandas as pd
import pickle as pkl
import h5py as h5

from dataloaders.base import BaseDataset


class CC3M(BaseDataset):
    name = 'cc3m'

    def __init__(self, config, writer, split, noise=False):
        super(CC3M, self).__init__(config, split)

        meta_path = self.config['meta_path']

        _, ext = os.path.splitext(meta_path)
        if ext.lower() == '.json':
            with open(meta_path) as fd:
                self.meta_data = json.loads(fd.read())
            # [video_id] = ['text', 'start', 'end', 'caption', 'sec']
        elif ext.lower() == '.pkl':
            with open(meta_path, 'rb') as fd:
                self.meta_data = pkl.load(fd)
        elif ext.lower() == '.csv':
        # ,url,start,end,caption,id,duration,download,video_path,unique_index
            self.meta_data = pd.read_csv(meta_path)

        self.sim_data = h5.File(self.config['hdf5_path'], 'r')

        if self.config['modality'] == 'both' or \
            'ensemble' in self.config['model_name']:
            self.text_embed_data = h5.File(self.config['text_embed_path'], 'r')
            self.vid_embed_data = h5.File(self.config['video_embed_path'], 'r')
        elif self.config['modality'] == 'text':
            self.text_embed_data = h5.File(self.config['text_embed_path'], 'r')
        elif self.config['modality'] == 'video':
            self.vid_embed_data = h5.File(self.config['video_embed_path'], 'r')

        if config['split_path']:
            self.meta_data = pd.read_pickle(config['split_path'])

        self.data_num  = len(self.meta_data)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        data       = self.meta_data.iloc[idx]
        real_text   = data['caption']
        start   = data['start']
        end   = data['end']
        orig_start = start #placeholder
        orig_end = end #placeholder
        unique_id   = data['unique_index']

        try:
            real_text_sim_matrix = np.array(self.sim_data[str(unique_id)])
        except Exception as e:
            print(e)
            real_text_sim_matrix = np.array([0,0,0,0])
            unique_id = -1

        if unique_id == -1:
            embedding = np.array([])
        elif self.config['modality'] == 'text':
            embedding = np.expand_dims(
                np.array(self.text_embed_data[str(unique_id)]), axis=0)
        elif self.config['modality'] == 'video':
            norm = np.linalg.norm(self.vid_embed_data[str(unique_id)])
            video_emb = self.vid_embed_data[str(unique_id)] / norm
            embedding = np.expand_dims(np.array(video_emb), axis=0)
        elif self.config['modality'] == 'both':
            norm = np.linalg.norm(self.vid_embed_data[str(unique_id)])
            video_emb = self.vid_embed_data[str(unique_id)] / norm
            concat_embedding = np.concatenate(
                [self.text_embed_data[str(unique_id)], video_emb])
            embedding = concat_embedding / np.linalg.norm(concat_embedding)

            embedding = np.expand_dims(embedding, axis=0)

        return (unique_id, real_text, start, end, orig_start,
                orig_end, real_text_sim_matrix, embedding)
