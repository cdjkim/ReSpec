from typing import Dict
from .base import BaseDataset
from .cc3m import CC3M
from .webvid import WebVid


DATASET: Dict[str, BaseDataset] = {
    WebVid.name: WebVid,
    CC3M.name: CC3M,
}
