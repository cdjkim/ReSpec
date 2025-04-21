import os
import argparse

import numpy as np
import pandas as pd

import colorful
from tqdm import tqdm

import h5py
import torch
from torch.utils.data import DataLoader

# For LanguageBind
from languagebind.languagebind import LanguageBind, LanguageBindImageTokenizer

# For dataset
from downstream_data.msrvtt      import MSRVTT
from downstream_data.didemo      import DiDeMo
from downstream_data.activitynet import ActivityNet
from downstream_data.lsmdc       import LSMDC
from downstream_data.youcook     import YouCook
from downstream_data.webvid      import WebVid
from downstream_data.cc3m        import CC3M

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275,  0.40821073)
OPENAI_DATASET_STD  = (0.26862954, 0.26130258, 0.27577711)

DATASET = {
    # Data Stream
    'webvid2m'   : WebVid,
    'cc3m'       : CC3M,

    # BT-Adapter retrieval
    'msrvtt'     : MSRVTT,
    'didemo'     : DiDeMo,
    'activitynet': ActivityNet,
    'lsmdc'      : LSMDC,
    'youcook'    : YouCook,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name',   type=str, required=True,    help='downstream dataset name')
    parser.add_argument('--split',       type=str, default ='train', help='[train/test/val]')

    parser.add_argument('--batch_size',  type=int, default=128,  help='batch size')
    parser.add_argument('--num_workers', type=int, default=8,    help='num workers')
    parser.add_argument('--max_frames',  type=int, default=8,    help='max frames')
    parser.add_argument('--max_words',   type=int, default=77,    help='max words')

    # TODO: save your path
    parser.add_argument('--msrvtt_video_root', type=str, 
                        default="path to raw video")
    parser.add_argument('--msrvtt_train_file', type=str, 
                        default="path to MSRVTT_train_9k.json")

    parser.add_argument('--didemo_video_root', type=str, 
                        default="path to raw video")
    parser.add_argument('--didemo_train_file', type=str, 
                        default="directory includes train_list.txt, train_data.json") 

    parser.add_argument('--activitynet_video_root', type=str,
                        default="path to raw video")
    parser.add_argument('--activitynet_train_file', type=str, 
                        default="directory includes anet_ret_train.json") 

    parser.add_argument('--youcook_video_root', type=str, 
                        default="path to raw video")
    parser.add_argument('--youcook_train_file', type=str, 
                        default="directory includes youcookii_annotations_trainval.json") 

    parser.add_argument('--lsmdc_video_root', type=str, 
                        default="path to raw video")
    parser.add_argument('--lsmdc_train_file', type=str, 
                        default="directory includes lsmdc_train.pkl") 

    parser.add_argument('--webvid2m_video_root', type=str, 
                        default="path to raw video")
    parser.add_argument('--webvid2m_train_file', type=str, 
                        default="path to results_2M_train_valid.pkl")

    parser.add_argument('--cc3m_video_root', type=str, 
                        default="path to raw video")
    parser.add_argument('--cc3m_train_file', type=str, 
                        default="path to video_cc_3m_final.csv")

    # TODO: save your path
    parser.add_argument('--save_path',   type=str, default='data', help='path to save embeddings')

    args = parser.parse_args()
    return args 

def get_LB_model(args):
    clip_type = {'video': 'LanguageBind_Video_FT',}  # also LanguageBind_Video
    cache_dir = './cache_dir'
    pt_ckpt   = 'LanguageBind/LanguageBind_Image'
    model     = LanguageBind(clip_type=clip_type, cache_dir=cache_dir)
    tokenizer = LanguageBindImageTokenizer.from_pretrained(
                        pt_ckpt, cache_dir=f'{cache_dir}/tokenizer_cache_dir')
    model.eval()
    return model, tokenizer

def get_h5py_files(args):
    h5 = {}
    save_root = os.path.join(args.save_path, args.data_name)
    os.makedirs(save_root, exist_ok=True)

    h5['text_emb_h5'] = h5py.File(os.path.join(save_root, f'text_emb.h5'), 'a')
    h5['clip_emb_h5'] = h5py.File(os.path.join(save_root, f'clip_emb.h5'), 'a')
    h5['clip_sim_h5'] = h5py.File(os.path.join(save_root, f'clip_sim.h5'), 'a')

    return h5

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Load model')
    model, tokenizer = get_LB_model(args)

    print('Load dataset/datalaoder')
    dataset  = DATASET[args.data_name](args)
    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        # collate_fn  = dataset.collate_fn,
        drop_last   = False,
        pin_memory  = True,  # Better when training on GPU.
        shuffle     = False
        ) # Don't need to shuffle for inference
    
    print('Set h5 file')
    h5 = get_h5py_files(args)

    print('Start downstream extraction')
    step = 0
    with torch.no_grad():
        model.eval()
        model.to(device)
        pbar = tqdm(dataloader)
        for batch in pbar:
            pbar.set_description(f"[{args.data_name}_{args.split}]")

            unique_ids, frames, raw_texts, valid_flag = batch
            if np.sum(np.array(valid_flag)) == 0:
                continue # ignore current batch when all clips are bad

            unique_ids = np.array(unique_ids)[np.array(valid_flag)]
            raw_texts  = list(np.array(raw_texts)[np.array(valid_flag)])
            texts      = tokenizer(raw_texts,
                                max_length=dataset.max_words,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt')

            inputs = {'video': {}}
            frames = frames[valid_flag].to(device)
            inputs['video']['pixel_values'] = frames

            inputs['language'] = texts
            texts['input_ids'] = texts['input_ids'].to(device)
            texts['attention_mask'] = texts['attention_mask'].to(device)

            embeddings = model(inputs)

            clip_emb   = embeddings['video'].detach().cpu().numpy()
            clip_emb   = clip_emb / np.linalg.norm(clip_emb, axis=1, keep_dims=True) # normalize
            text_emb   = embeddings['language'].detach().cpu().numpy()
            similarity = np.sum(clip_emb * text_emb, axis=1) 

            for u_id, c_emb, t_emb, sim in zip(unique_ids, clip_emb, text_emb, similarity):
                if args.data_name == 'webvid2m':
                    group = str(int(u_id) // 100000)
                    if group in h5['text_emb_h5'].keys():
                        group_text = h5['text_emb_h5'][group]
                        group_clip = h5['clip_emb_h5'][group]
                        group_sim  = h5['clip_sim_h5'][group]
                    else:
                        group_text = h5['text_emb_h5'].create_group(group)
                        group_clip = h5['clip_emb_h5'].create_group(group)
                        group_sim  = h5['clip_sim_h5'].create_group(group)
                    group_text.create_dataset(str(u_id), data = t_emb)
                    group_clip.create_dataset(str(u_id), data = c_emb)
                    group_sim.create_dataset(str(u_id), data = np.array([[sim]]))

                else:
                    h5['text_emb_h5'].create_dataset(str(u_id), data = t_emb)
                    h5['clip_emb_h5'].create_dataset(str(u_id), data = c_emb)
                    h5['clip_sim_h5'].create_dataset(str(u_id), data = np.array([[sim]]))
            step += 1

    for key in h5.keys():
        h5[key].flush()
        h5[key].close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print(colorful.bold_pink('Thank you and Good Job Computer.').styled_string)