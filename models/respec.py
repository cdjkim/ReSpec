from abc import ABC, abstractmethod

import os
import sys
import numpy as np
import pickle as pkl
import torch

from tqdm import tqdm

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from .online_filter import OnlineFilter


class ReSpec(OnlineFilter):
    def __init__(self, config, scheduler, writer):
        super().__init__(config, scheduler, writer)
        self.data_name = self.config['data_name']
        self.encoder   = self.config['encoder']
        self.data_size = 0

        self.threshold = self.config['threshold']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # downstream LB embedding extraction
        # set downstream tasks

        self.relevance_modality = config['relevance_modality']
        assert self.relevance_modality in ['text', 'video', 'both']

        per_task_vmfkde_path = 'data/per_task_vmf_kde'
        self.downstream_tasks = ['msrvtt', 'didemo', 'activitynet', 'youcook', 'lsmdc']

        print('DOWNSTREAM TASKS: ', self.downstream_tasks)

        task2estimator = {}
        task2density_threshold = {}
        for task_name in self.downstream_tasks:
            meta_path = os.path.join(per_task_vmfkde_path, f'{task_name}', 'vmf_kde_stats.pkl')

            with open(meta_path, 'rb') as f:
                task_meta = pkl.load(f)
            task2estimator[task_name] = {'text': task_meta['text_kde_estimator'],
                                            'video': task_meta['video_kde_estimator']}
            task2density_threshold[task_name] = {}
            for m in ['text', 'video']:
                task2density_threshold[task_name][m] = np.percentile(task_meta[f'{m}_densities'], 5)

        self.task2estimator = task2estimator
        self.task2density_threshold = task2density_threshold

        self.root_embedding = torch.load(f'data/text_root_embedding.pt',
                                            map_location='cpu').detach().to(self.device)

        self.dr_threshold = float(config['dr_txt_thr'])
        self.task2dr_threshold = {}

        for task_name in task2estimator.keys():
            with torch.no_grad():
                # euclidean distance of downstream embeddings from the root
                task_drs = torch.norm(task2estimator[task_name]['text'].data - self.root_embedding, dim=-1).cpu().numpy()

            self.task2dr_threshold[task_name] = np.percentile(task_drs, float(config['dr_txt_thr']))

        self.cnt_sampled = 0
        self.sampled_in = []

    @torch.no_grad()
    def filter(self):
        def check_relevant_clean_specific(task_name, embedding, s):
            filter_pass = {
                'relevant': False,
                'clean': False,
                'specific': False,
            }
            modal_densities = self.task2estimator[task_name][self.relevance_modality].density(embedding).cpu().numpy()
            if modal_densities[0] >= self.task2density_threshold[task_name][self.relevance_modality]:
                filter_pass['relevant'] = True
                if s > self.threshold:
                    filter_pass['clean'] = True
                    if self.config['modality'] == 'text':
                        # check specific
                        dr = torch.norm(embedding - self.root_embedding, dim=-1)
                        if dr.item() > self.task2dr_threshold[task_name]:
                            filter_pass['specific'] = True

                return filter_pass
            else:
                return filter_pass

        for vid_data in tqdm(self.dataloader):

            video_id, real_text, start, end, orig_start, orig_end, \
                real_text_sim, embedding= vid_data
            real_text_sim = real_text_sim[0] # clip_id x text_id
            video_id = video_id[0]
            real_text = real_text[0]
            start = start[0]
            end = end[0]
            orig_start = orig_start[0]
            orig_end = orig_end[0]
            embedding = embedding[0]

            if video_id == -1:
                continue

            if isinstance(embedding, np.ndarray):
                embedding = torch.from_numpy(embedding)
            embedding = embedding.to(self.device)

            s = float(real_text_sim)
            for task_name in self.downstream_tasks:
                filter_pass = check_relevant_clean_specific(task_name, embedding, s)
                if filter_pass['clean'] and filter_pass['relevant'] and filter_pass['specific']:
                    _ = self.store_result(video_id, real_text, start,
                                        end, orig_start, orig_end)

            embedding = embedding.cpu().numpy()

        # save fig
        save_path = self.save_result()
        self.gen_downstream_cmd(save_path)
