## Feature Extractor

Set up the code and environment following the official LanguageBind repository.
- [Official LanguageBind repo](https://github.com/PKU-YuanGroup/LanguageBind)

```
conda create -n LB_feature_extraction python=3.9
conda activate LB_feature_extraction
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

**Copy all files from the current directory to the root directory of the official LanguageBind repository.**

## Data
Refer to the given GitHub repository to set up the datasets for each pre-train and downstream task:
- [WebVid2M](https://github.com/m-bain/webvid)
- [VideoCC3M](https://github.com/google-research-datasets/videoCC-data)
- [MSR-VTT retrieval](https://github.com/ArrowLuo/CLIP4Clip)
- [DiDeMo retrieval](https://github.com/ArrowLuo/CLIP4Clip)
- [ActivityNet retrieval](https://github.com/ArrowLuo/CLIP4Clip)
- [YouCook retrieval](http://youcook2.eecs.umich.edu)
- [LSMDC retrieval](https://github.com/ArrowLuo/CLIP4Clip)

Then, configure `{data_name}_video_root` and `{data_name}_train_path` in `feature_extraction.py`.

- Pre-extract pre-train / downstream feature embeddings
    ```
    python feature_extraction.py --data_name {data_name} --save_path {save_path}
    ```

- Pre-extract root embeddings
    ```
    python root_embedding_extraction.py --save_path {save_path}
    ```
