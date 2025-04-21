# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Callable, List, Optional, Union, Any

import torch
from mmengine.dataset import BaseDataset

from mmaction.utils import ConfigType


class BaseActionDataset(BaseDataset, metaclass=ABCMeta):
    """Base class for datasets.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict, optional): Path to a directory where
            videos are held. Defaults to None.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Defaults to False.
        num_classes (int, optional): Number of classes of the dataset, used in
            multi-class datasets. Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Defaults to 0.
        modality (str): Modality of data. Support ``RGB``, ``Flow``, ``Pose``,
            ``Audio``. Defaults to ``RGB``.
    """

    def __init__(self,
                #  ann_file: str,
                 ann_file,
                 pipeline: List[Union[ConfigType, Callable]],
                 data_prefix: Optional[ConfigType] = dict(prefix=''),
                 test_mode: bool = False,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 0,
                 modality: str = 'RGB',
                 **kwargs) -> None:
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        if 'random_sample' in kwargs:
            self.random_sample = kwargs.pop('random_sample')
        if 'split_file' in kwargs:
            self.split_file = kwargs.pop('split_file')
        if 'random_seed' in kwargs:
            self.random_seed = kwargs.pop('random_seed')
        if type(ann_file) is tuple:
            self.ann_file_cc3m      = ann_file[0]
            self.ann_file_webvid2m  = ann_file[1]
            data_root = kwargs.pop('data_root')
            self.data_root_cc3m     = data_root[0]
            self.data_root_webvid2m = data_root[1]
            ann_file  = ann_file[0] # dummy
            data_root = data_root[0] # dummy

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        data_info['start_index'] = self.start_index

        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[data_info['label']] = 1.
            data_info['label'] = onehot

        return data_info

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """

        data_info = self.get_data_info(idx)
        #data = self.pipeline(data_info)
        try:
            data = self.pipeline(data_info)
        except Exception as e:
            print (e)
            data = None
        return data
