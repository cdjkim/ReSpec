# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmaction.registry import METRICS
from mmaction.models.heads.mug_head import Mug_head, DRL_head
from mmaction.evaluation import top_k_accuracy
import pandas as pd
import json
import math

import getpass
import socket
USERNAME = getpass.getuser()
IPADDR   = socket.gethostbyname(socket.gethostname())

@METRICS.register_module()
class RetrievalMetric(BaseMetric):
    """Metric for video retrieval task.

    Args:
        metric_list (str | tuple[str]): The list of the metrics to be
            computed. Defaults to ``('R1', 'R5', 'R10', 'MdR', 'MnR')``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix = 'retrieval'

    def __init__(self,
                 metric_list: Union[Tuple[str],
                                    str] = ('R1', 'R5', 'R10', 'MdR', 'MnR'),
                 collect_device: str = 'gpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if isinstance(metric_list, str):
            metric_list = (metric_list, )

        for metric in metric_list:
            if metric not in ['R1', 'R5', 'R10', 'MdR', 'MnR']:
                raise ValueError(f'RetrievalMetric only supports '
                                 f"'R1', 'R5', 'R10', 'MdR', 'MnR', "
                                 f"but got '{metric}. '")

        self.metric_list = metric_list

    def process(self, data_batch: Optional[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict, optional): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)

        for data_sample in data_samples:
            results = dict()
            features = data_sample['features']
            video_feature = features['video_feature'].cpu().numpy()
            text_feature = features['text_feature'].cpu().numpy()
            results['video_feature'] = video_feature
            results['text_feature'] = text_feature
            self.results.append(results)


    def compute_metrics(self, results: List) -> Dict:
        video_features = np.stack([res['video_feature'] for res in results])
        text_features = np.stack([res['text_feature'] for res in results])

        video_features = video_features / np.linalg.norm(
            video_features, axis=-1, keepdims=True)
        text_features = text_features / np.linalg.norm(
            text_features, axis=-1, keepdims=True)

        similarity = text_features @ video_features.T

        sx = np.sort(-similarity)
        d = np.diag(-similarity)
        ind = np.where((sx - d[:, None]) == 0)[1]

        metrics = OrderedDict()
        for metric in self.metric_list:
            if metric == 'R1':
                metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
            elif metric == 'R5':
                metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
            elif metric == 'R10':
                metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
            elif metric == 'MdR':
                metrics['MdR'] = np.median(ind) + 1
            elif metric == 'MnR':
                metrics['MnR'] = np.mean(ind) + 1

        return metrics


@METRICS.register_module()
class VttQAMetric(BaseMetric):
    """Metric for video retrieval task.

    Args:
        metric_list (str | tuple[str]): The list of the metrics to be
            computed. Defaults to ``('R1', 'R5', 'R10', 'MdR', 'MnR')``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix = 'retrieval'

    def __init__(self,
                 metric_list: Union[Tuple[str],
                                    str] = ('Acc'),
                 collect_device: str = 'gpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if isinstance(metric_list, str):
            metric_list = (metric_list, )

        for metric in metric_list:
            if metric not in ['Acc']:
                raise ValueError(f'VttQAMetric only supports '
                                 f"'Acc', "
                                 f"but got '{metric}. '")

        self.metric_list = metric_list

    def process(self, data_batch: Optional[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict, optional): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)

        for data_sample in data_samples:
            results = dict()
            features = data_sample['features']
            video_feature = features['video_feature'].cpu().numpy()
            text_feature = features['text_feature'].cpu().numpy()
            results['video_feature'] = video_feature
            results['text_feature'] = text_feature
            self.results.append(results)


    def compute_metrics(self, results: List) -> Dict:
        video_features = np.stack([res['video_feature'] for res in results])
        text_features = np.stack([res['text_feature'] for res in results])
        csv_data = pd.read_csv('/gallery_mfa/sangwoo.moon/data/video/MSRVTT/MSR_MC_test.csv', sep="\t")
        labels = np.array(csv_data["answer"].values)

        video_features = video_features / np.linalg.norm(
            video_features, axis=-1, keepdims=True)
        text_features = text_features / np.linalg.norm(
            text_features, axis=-1, keepdims=True)

        similarity = np.sum(video_features * text_features, axis=-1)
        preds = np.argmax(similarity.reshape(-1, 5), axis=-1)

        metrics = OrderedDict()
        metrics['Acc'] = sum(preds == labels) / len(labels)

        return metrics


@METRICS.register_module()
class How2QAMetric(BaseMetric):
    """Metric for video retrieval task.

    Args:
        metric_list (str | tuple[str]): The list of the metrics to be
            computed. Defaults to ``('R1', 'R5', 'R10', 'MdR', 'MnR')``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix = 'retrieval'

    def __init__(self,
                 metric_list: Union[Tuple[str],
                                    str] = ('Acc'),
                 collect_device: str = 'gpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if isinstance(metric_list, str):
            metric_list = (metric_list, )

        for metric in metric_list:
            if metric not in ['Acc']:
                raise ValueError(f'How2QAMetric only supports '
                                 f"'Acc', "
                                 f"but got '{metric}. '")

        self.metric_list = metric_list

    def process(self, data_batch: Optional[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict, optional): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)

        for data_sample in data_samples:
            results = dict()
            features = data_sample['features']
            video_feature = features['video_feature'].cpu().numpy()
            text_feature = features['text_feature'].cpu().numpy()
            results['video_feature'] = video_feature
            results['text_feature'] = text_feature
            self.results.append(results)


    def compute_metrics(self, results: List) -> Dict:
        video_features = np.stack([res['video_feature'] for res in results])
        text_features = np.stack([res['text_feature'] for res in results])

        video_features = video_features / np.linalg.norm(
            video_features, axis=-1, keepdims=True)
        text_features = text_features / np.linalg.norm(
            text_features, axis=-1, keepdims=True)

        similarity = np.sum(video_features * text_features, axis=-1)
        preds = np.argmax(similarity.reshape(-1, 4), axis=-1)
        labels = np.zeros_like(preds)

        metrics = OrderedDict()
        metrics['Acc'] = sum(preds == labels) / len(labels)

        return metrics


@METRICS.register_module()
class HowToAlignMetric(BaseMetric):
    """Metric for video retrieval task.

    Args:
        metric_list (str | tuple[str]): The list of the metrics to be
            computed. Defaults to ``('R1', 'R5', 'R10', 'MdR', 'MnR')``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix = 'retrieval'

    def __init__(self,
                 metric_list: Union[Tuple[str],
                                    str] = ('R@1', 'AUC'),
                 collect_device: str = 'gpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if isinstance(metric_list, str):
            metric_list = (metric_list, )

        for metric in metric_list:
            if metric not in ['R@1', 'AUC']:
                raise ValueError(f'VttQAMetric only supports '
                                 f"'R@1', 'AUC' "
                                 f"but got '{metric}. '")

        self.metric_list = metric_list

    def process(self, data_batch: Optional[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict, optional): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)

        for data_sample in data_samples:
            results = dict()
            features = data_sample['features']
            video_feature = features['video_feature'].cpu().numpy()
            text_feature = features['text_feature'].cpu().numpy()
            results['video_feature'] = video_feature
            results['text_feature'] = text_feature
            self.results.append(results)


    def compute_metrics(self, results: List) -> Dict:
        video_features = np.stack([res['video_feature'] for res in results])
        text_features = np.stack([res['text_feature'] for res in results])
        video_features = video_features / np.linalg.norm(
            video_features, axis=-1, keepdims=True)
        text_features = text_features / np.linalg.norm(
            text_features, axis=-1, keepdims=True)

        with open('/gallery_tate/dongyeon.woo/howto100m/howtoalign/raw_clips/htm_align_data.json', 'r') as fp:
            video_data = json.load(fp)
        video_data = [data for key, data in video_data.items()]

        recall = []
        total_vlen = []
        total_text_count = []
        total_aligned_count = []

        total_align_sim = []
        total_align_tgt = []

        seq_len = 32
        method = 'overlap-seq'  # 'overlap-seq' or 'global'
        print(f'Test Alignment with {method} method')
        device = torch.device('cpu')

        start_index = 0
        for data in video_data[:2]:
            text_str = data['text_str']
            tgt_aligned = data['tgt_aligned']
            data['start'] = np.array(data['start'])
            data['end'] = np.array(data['end'])
            vlen = data['vlen']

            text_str_aligned = np.array(text_str)[np.array(tgt_aligned).astype(bool)].tolist()
            start_idx_aligned = data['start'][np.array(tgt_aligned).astype(bool)]
            end_idx_aligned = data['end'][np.array(tgt_aligned).astype(bool)]

            eps = torch.tensor(1e-5, device=device)
            step = np.arange(0, vlen - seq_len // 2, seq_len // 4)

            interpolate_text_mid_ts = data['start'] + data['end'] / 2

            logits = torch.zeros(len(text_str), vlen, device=device)
            logits_dual = torch.zeros(len(text_str), vlen, device=device)
            overlap_counter = torch.zeros(len(text_str), vlen, device=device)
            logits_a_dual = torch.zeros(len(text_str), device=device)
            logits_a_joint = torch.zeros(len(text_str), device=device)
            text_overlap_counter = torch.zeros(len(text_str), device=device)

            for idx, step_ in enumerate(step):
                nonalignable_text_idx = np.arange(len(text_str))[~np.array(tgt_aligned).astype(bool)]
                nonalignable_text_mid_ts = interpolate_text_mid_ts[~np.array(tgt_aligned).astype(bool)]
                nonalignable_text_window_mask = np.logical_and(
                    step_ - seq_len <= nonalignable_text_mid_ts,
                    nonalignable_text_mid_ts <= step_ + seq_len + seq_len)
                active_nonalignable_text_idx = nonalignable_text_idx[nonalignable_text_window_mask]
                if len(active_nonalignable_text_idx) == 0:
                    continue

                text_window_left, text_window_right = (
                    active_nonalignable_text_idx.min(),
                    active_nonalignable_text_idx.max())
                active_text_mask = np.zeros((len(text_str))).astype(bool)
                # handle edge case, otherwise the heading and tailing alignable texts could be missed
                if idx <= 3:
                    text_window_left = 0
                elif idx >= len(step) - 4:
                    text_window_right = vlen
                active_text_mask[text_window_left: text_window_right + 1] = True

                active_text_str = np.array(text_str)[active_text_mask].tolist()
                active_text_mask_tensor = torch.from_numpy(active_text_mask).to(device).bool()

                if np.sum(active_text_mask) == 0:
                    continue

                logits_ = {}
                v = torch.Tensor(video_features[start_index : start_index + len(step), :])
                t = torch.Tensor(text_features[start_index + text_window_left : start_index + text_window_right + 1, :])
                similarity = torch.matmul(v, t.transpose(0, 1)).transpose(-1, -2).unsqueeze(0)
                logits_['sim'] = similarity
                logits_['dual-sim'] = similarity

                logits_a_dual_ = logits_['dual-sim'][0].max(-1).values
                logits_a_joint_ = logits_['sim'][0].max(-1).values

                try:
                    logits_a_dual[active_text_mask_tensor] += logits_a_dual_
                except:
                    from IPython import embed; embed(colors='neutral')  # XXX DEBUG  # yapf: disable
                logits_a_joint[active_text_mask_tensor] += logits_a_joint_
                text_overlap_counter[active_text_mask_tensor] += 1

                logits[active_text_mask_tensor, step_:min(vlen, step_ + seq_len)] += logits_['sim'][0, :, :min(vlen, step_ + seq_len) - step_]
                logits_dual[active_text_mask_tensor, step_:min(vlen, step_ + seq_len)] += logits_['dual-sim'][0, :,
                                                                                        :min(vlen, step_ + seq_len) - step_]
                overlap_counter[active_text_mask_tensor, step_:min(vlen, step_ + seq_len)] += 1
            logits = logits.div(torch.maximum(overlap_counter, eps))
            logits_dual = logits_dual.div(torch.maximum(overlap_counter, eps))

            logits_a_dual = logits_a_dual.div(torch.maximum(text_overlap_counter, eps))
            logits_a_joint = logits_a_joint.div(torch.maximum(text_overlap_counter, eps))
            sim = (logits + logits_dual) / 2

            sim.masked_fill_(sim == 0, -6e4)
            prob = sim.softmax(-1)
            vlen = sim.size(-1)

            total_align_tgt.append(np.array(tgt_aligned))
            total_align_sim.append(sim.max(-1)[0].cpu().numpy())

            sim = sim[torch.as_tensor(tgt_aligned).bool(), :]
            prob = prob[torch.as_tensor(tgt_aligned).bool(), :]

            for text_idx in range(sim.size(0)):
                s = math.floor(start_idx_aligned[text_idx])
                e = math.ceil(end_idx_aligned[text_idx])
                recall.append(s <= prob[text_idx].argmax(-1).item() <= e)

            total_vlen.append(vlen)
            total_text_count.append(len(text_str))
            total_aligned_count.append(len(text_str_aligned))

            start_index += max(len(step), len(text_str))

        total_align_sim = np.concatenate(total_align_sim, 0)
        total_align_tgt = np.concatenate(total_align_tgt, 0)
        assert total_align_tgt.shape == total_align_sim.shape

        auc = metrics.roc_auc_score(total_align_tgt, total_align_sim)

        similarity = np.sum(video_features * text_features, axis=-1)
        preds = np.argmax(similarity.reshape(-1, 5), axis=-1)

        metrics = OrderedDict()
        metrics['R@1'] = np.mean(recall)
        metrics['AUC'] = auc

        return metrics


@METRICS.register_module()
class PostProc_RetrievalMetric(BaseMetric):
    """Metric for video retrieval task.

    Args:
        metric_list (str | tuple[str]): The list of the metrics to be
            computed. Defaults to ``('R1', 'R5', 'R10', 'MdR', 'MnR')``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix = 'retrieval'

    def __init__(self,
                 metric_list: Union[Tuple[str],
                                    str] = ('R1', 'R5', 'R10', 'MdR', 'MnR'),
                 collect_device: str = 'gpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if isinstance(metric_list, str):
            metric_list = (metric_list, )

        for metric in metric_list:
            if metric not in ['R1', 'R5', 'R10', 'MdR', 'MnR']:
                raise ValueError(f'RetrievalMetric only supports '
                                 f"'R1', 'R5', 'R10', 'MdR', 'MnR', "
                                 f"but got '{metric}. '")
        self.Mug_head = Mug_head()
        self.Mug_head.eval()
        self.metric_list = metric_list

    def process(self, data_batch: Optional[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict, optional): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)

        for data_sample in data_samples:
            results = dict()
            features = data_sample['features']
            video_feature = features['video_feature'].cpu().numpy()
            text_feature = features['text_feature'].cpu().numpy()
            mask = features['mask'].cpu().numpy()
            results['video_feature'] = video_feature
            results['mask'] = mask
            results['text_feature'] = text_feature
            self.results.append(results)


    def compute_metrics(self, results: List) -> Dict:
        video_features = np.stack([res['video_feature'] for res in results])
        text_features = np.stack([res['text_feature'] for res in results])
        mask = np.stack([res['mask'] for res in results])

        similarity = self.Mug_head(torch.from_numpy(text_features), torch.from_numpy(video_features), torch.from_numpy(mask))
        similarity = similarity.numpy()

        sx = np.sort(-similarity)
        d = np.diag(-similarity)
        ind = np.where((sx - d[:, None]) == 0)[1]

        metrics = OrderedDict()
        for metric in self.metric_list:
            if metric == 'R1':
                metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
            elif metric == 'R5':
                metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
            elif metric == 'R10':
                metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
            elif metric == 'MdR':
                metrics['MdR'] = np.median(ind) + 1
            elif metric == 'MnR':
                metrics['MnR'] = np.mean(ind) + 1

        return metrics

@METRICS.register_module()
class ZeroShotAccMetric(BaseMetric):
    """Metric for video retrieval task.

    Args:
        metric_list (str | tuple[str]): The list of the metrics to be
            computed. Defaults to ``('R1', 'R5', 'R10', 'MdR', 'MnR')``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix = 'retrieval'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = (
                     'top_k_accuracy', 'mean_class_accuracy'),
                 collect_device: str = 'cpu',
                 metric_options: Optional[Dict] = dict(
                     top_k_accuracy=dict(topk=(1, 5))),
                 prefix: Optional[str] = None) -> None:

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in [
                'top_k_accuracy', 'mean_class_accuracy',
                'mmit_mean_average_precision', 'mean_average_precision'
            ]

        self.metrics = metrics
        self.metric_options = metric_options

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            label = data_sample['gt_labels']
            features = data_sample['features']
            video_feature = features['video_feature'].cpu().numpy()
            text_feature = features['text_feature'].cpu().numpy()
            result['video_feature'] = video_feature
            if not hasattr(self,"text_feature"):
                self.text_feature = text_feature

            if 'mask' in features:
                if not hasattr(self,"mask"):
                    self.mask = features['mask'].cpu().numpy()

            if label['item'].size(0) == 1:
                # single-label
                result['label'] = label['item'].item()
            else:
                # multi-label
                result['label'] = label['item'].cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        video_features = np.stack([res['video_feature'] for res in results])
        text_features = self.text_feature
        labels = [x['label'] for x in results]

        if hasattr(self, 'mask'):
            mask = self.mask
            head = Mug_head()
            score = head(torch.from_numpy(text_features), torch.from_numpy(video_features), torch.from_numpy(mask)).numpy()
            score = score.T

        else:
            video_features = video_features / np.linalg.norm(
                video_features, axis=-1, keepdims=True)
            text_features = text_features / np.linalg.norm(
                text_features, axis=-1, keepdims=True)
            score = video_features @ text_features.T

        top_k_acc = top_k_accuracy(score, labels, (1,5))
        metrics = {}
        metrics['overall_acc1'] = top_k_acc[0]
        metrics['overall_acc5'] = top_k_acc[1]

        return metrics
