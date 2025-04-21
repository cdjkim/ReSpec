import colorful

from torch.utils.data import ConcatDataset, DataLoader

from dataloaders import DATASET

# =========
# Scheduler
# =========
class DataScheduler():
    def __init__(self, config, writer):
        self.config = config
        self.datasets = {}
        self.total_step = 0

        # Prepare datasets
        dataset_name = config['data_name']

        self.datasets[dataset_name] = DATASET[dataset_name](self.config, writer, split='train', noise=True)

        self.total_step += len(self.datasets[dataset_name]) // self.config['batch_size']

        self.task_datasets = []

        dataset = ConcatDataset([self.datasets[dataset_name]])
        self.task_datasets.append((1, dataset))

    def __iter__(self):
        for t_i, (epoch, task) in enumerate(self.task_datasets):
            print(colorful.bold_green('\nProgress to Task %d' % t_i).styled_string)
            collate_fn = task.datasets[0].dataset.collate_fn
            # for data in DataLoader(task, batch_size=self.config['batch_size'],
            #     num_workers=self.config['num_workers'],
            #     collate_fn=collate_fn, drop_last=True): # shuffle=True
            #     yield data, t_i
            task_loader = DataLoader(task, batch_size=self.config['batch_size'],
                            num_workers=self.config['num_workers'],
                            collate_fn=collate_fn,
                            drop_last=False,
                            pin_memory=True, # better when training on GPU.
                            shuffle=True)

            yield task_loader, epoch, t_i

    def get_task(self, t):
        return self.task_datasets[t][1]

    def get_dataloader(self, dataset):
        collate_fn = dataset.dataset.datasets[0].dataset.collate_fn
        return DataLoader(dataset, self.config['batch_size'], shuffle=True, collate_fn=collate_fn)

    def __len__(self):
        return self.total_step
