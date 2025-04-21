from abc import ABC, abstractmethod
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader

# ================
# Generic Datasets
# ================
class BaseDataset(Dataset, ABC):
    name = 'base'
    dataset_size = 0

    def __init__(self, config, split):
        super().__init__()
        self.config = config
        self.subsets = dict()
        self.split = split

    def __len__(self):
        return self.dataset_size

    def eval(self, model, writer: SummaryWriter, step, t=None, task_labels=None, eval_title=''):
        if self.config['eval']:
            self._eval_model(model, writer, step, t, task_labels, eval_title)

    # @abstractmethod
    def _eval_model(self, model, writer: SummaryWriter, step, eval_title):
        raise NotImplementedError

    def collate_fn(self, batch):
        return default_collate(batch)

