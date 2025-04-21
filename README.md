# **ReSpec**: **Re**levance and **Spec**ificity Grounded Online Filtering for Learning on Video-Text Data Streams

This repository contains the code for our CVPR 2025 paper.

- Chris Dongjoo Kim*, Jihwan Moon*, Sangwoo Moon, Heeseung Yun, Sihaeng Lee, Aniruddha Kembhavi, Soonyoung Lee, Gunhee kim, Sangho Lee, Christopher Clark. ReSpec: Relevance and Specificity Grounded Online Filtering for Learning on Video-Text Data Streams. In CVPR, 2025 (* equal contribution).

## System Dependencies
- Python >= 3.9
- CUDA >= 9.0 supported GPU

## Installation
Using virtual env is recommended.
```bash
# create conda env with python=3.9
conda create -n {ENV_NAME} python=3.9
conda activate {ENV_NAME}
# install other packages
pip install -r requirements.txt
```

## Preparation
### Data and Log directory set-up
create `checkpoints` and `data` directories.

```bash
$ mkdir -p data/videocc3m
$ mkdir -p data/webvid2m
```

### Pre-extract similarity and LB feature embedding

Perform extraction referring to the [LB_feature_extraction](LB_feature_extraction) directory.

### Kernel density estimator

For each task (e.g., `msrvtt, didemo, lsmdc, youcook, activitynet`) , run command below to save von Mises-Fisher (VMF) kernel density estimator (KDE)

```bash
$ python -m utils.vmf_kde_each_task --task {task} --save_path {per task vmf kde path} --embedding_path {save_path from LB_feature_extraction}
```

Then, create symbolic link to

```bash
$ ln -s [per task vmf kde path] data/per_task_vmf_kde
```

## Filtering

Perform ReSpec filtering:
`threshold` uses normalized values between [0,1].

```bash
$ python main.py -c=configs/{cc3m/webvid}_respec.yaml -l=checkpoints/{checkpoint directory} --override="model_name=respec|threshold={sim_thr}"
```
Obtain the path to the saved `clean_meta_real_text_sim.pkl` file containing the filtered meta data, and train on [BT-adapter](BT-adapter).


## Citation
The contents of this repo are free to use for academic purposes only. If you use any of the material in this repository as part of your work, we ask you to cite:

```
@inproceedings{respec-CVPR-2025,
    author    = {Chris Dongjoo Kim and Jihwan Moon and Sangwoo Moon and Heeseung Yun and Sihaeng Lee and Aniruddha Kembhavi and Soonyoung Lee and Gunhee kim and Sangho Lee and Christopher Clark},
    title     = "{ReSpec: Relevance and Specificity Grounded Online Filtering for Learning on Video-Text Data Streams}"
    booktitle = {CVPR},
    year      = 2025
}
```
