# BT-Adapter
This repository is used for video-text pretraining and downstream video task evaluation.

## Getting Started

### Installation

Below is a clunky installation of the original BT-Adapter for our system.
Please refer to the [original BT-adapter](https://github.com/farewellthree/BT-Adapter/tree/main/mmaction2) set-up suitable your system.
```bash
conda create --name btadapter_mmaction python=3.10
conda activate btadapter_mmaction
bash install.sh
```

### Prepare Datasets
## Data
Refer to the given GitHub repository to set up the datasets for each pre-train and downstream task:
- [WebVid2M](https://github.com/m-bain/webvid)
- [VideoCC3M](https://github.com/google-research-datasets/videoCC-data)

update `configs/btadapter/pretrain/base/webvid.py` line 34-37 with you data paths

update `configs/btadapter/pretrain/base/cc3m.py` line 35-38 with your data paths

update the evaluation data paths in `configs/btadapter/zero-shot/{lsmdc/activitynet/didemo/msrvtt/youcook}_btadapterl14.py`

## Train and Eval
```bash
torchrun --nproc_per_node=4 --master_port=54582 tools/train.py configs/btadapter/pretrain/base/webvid.py --cfg-options work_dir=[path to clean_meta_real_text_sim.pkl] train_epoch=1 --no-validate --seed [random_seed]
```
