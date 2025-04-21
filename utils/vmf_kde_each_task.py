import os
import argparse
import pickle
import torch
import numpy as np
import h5py
import einops
from tqdm import tqdm
import json
import pickle as pkl
from collections import defaultdict

import sys
from utils.vmf_kde import VonMisesFisherKDE


TASKS = ['activitynet', 'didemo', 'lsmdc', 'msrvtt', 'youcook']

def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

@torch.no_grad()
def main(args):

    save_dir = os.path.join(args.save_path, args.task)

    text_embeddings = []
    video_embeddings = []
    task = args.task
    print(f"Processing {task}")

    task_text_embs = h5py.File(os.path.join(args.embedding_path, task, 'text_emb.h5'), 'r')
    task_video_embs = h5py.File(os.path.join(args.embedding_path, task, 'clip_emb.h5'), 'r')
    for key in tqdm(task_text_embs):
        text_embeddings.append(task_text_embs[key])
        task_video_emb = normalize(task_video_embs[key])
        video_embeddings.append(task_video_emb)

    text_embeddings = np.array(text_embeddings)
    video_embeddings = np.array(video_embeddings)

    # normalize again, just in case
    text_embeddings = normalize(text_embeddings)
    video_embeddings = normalize(video_embeddings)

    print(f"Processing Complete: {len(text_embeddings)} Embeddings")
    print(f"Embedding shape: {text_embeddings.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_embeddings = torch.from_numpy(text_embeddings).to(device)
    video_embeddings = torch.from_numpy(video_embeddings).to(device)

    text_kde_estimator = VonMisesFisherKDE(text_embeddings, device=device)
    print(f"text KDE estimator Kappa: {text_kde_estimator.kappa}")

    video_kde_estimator = VonMisesFisherKDE(video_embeddings, device=device)
    print(f"video KDE estimator Kappa: {video_kde_estimator.kappa}")

    embeddings = {'text': text_embeddings,
                  'video': video_embeddings,}
    kde_estimator = {'text': text_kde_estimator,
                     'video': video_kde_estimator}

    # only used for analysis.
    density_stats = {}
    for m in embeddings.keys():
        estimated_densities = []
        with torch.no_grad():
            for idx in tqdm(range(len(embeddings[m]))):
                embedding = embeddings[m][idx].unsqueeze(0)
                density = kde_estimator[m].density(embedding, exclude_idx=idx).cpu().numpy()  # exclude itself to obtain "unbiased" density estimate
                estimated_densities.append(density)
        estimated_densities = np.concatenate(estimated_densities, axis=0)

        density_stats[f'{m}_densities'] = estimated_densities

    os.makedirs(f'{save_dir}', exist_ok=True)
    with open(os.path.join(save_dir, f'vmf_kde_stats.pkl'), 'wb') as fw:
        d = {**density_stats,
             'text_kde_estimator': text_kde_estimator,
             'video_kde_estimator': video_kde_estimator,}
        pickle.dump(d, fw)

    return density_stats

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--embedding_path", required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    density_stats = main(args)
