#%% Imports
import numpy as np
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from experiments.dissentaglement.legacy.gendata import sample_video


#%% On-the-fly simulator dataset
class BinaryShapesVideoDataset(IterableDataset):
    """Calls ``gendata.sample_video`` for every item; nothing is stored on disk."""

    def __init__(self, sim_config, samples_per_epoch, seed, fixed_parameters=None):
        self.sim_config = dict(sim_config)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.fixed_parameters = {} if fixed_parameters is None else dict(fixed_parameters)
        self.epoch = 0
        self.max_val = 1.0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.samples_per_epoch

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch + 9_973 * worker_id)

        for _ in range(worker_id, self.samples_per_epoch, num_workers):
            yield sample_video(rng, self.sim_config, fixed=self.fixed_parameters)


def numpy_collate(batch):
    videos = np.stack([item[0] for item in batch]).astype(np.float32)
    parameters = np.stack([item[1] for item in batch]).astype(np.float32)
    return videos, parameters


#%% Public loader factory

def get_dataloaders(config, fixed_parameters=None):
    data_cfg = config["data"]
    sim_cfg = config["simulation"]
    seed = int(config["seed"])

    train_dataset = BinaryShapesVideoDataset(
        sim_cfg,
        samples_per_epoch=data_cfg["train_samples_per_epoch"],
        seed=seed,
        fixed_parameters=fixed_parameters,
    )
    test_dataset = BinaryShapesVideoDataset(
        sim_cfg,
        samples_per_epoch=data_cfg["eval_samples"],
        seed=seed + 100_000,
        fixed_parameters=fixed_parameters,
    )

    if config.get("debug", False):
        train_dataset.samples_per_epoch = min(train_dataset.samples_per_epoch, 2 * data_cfg["batch_size"])
        test_dataset.samples_per_epoch = min(test_dataset.samples_per_epoch, data_cfg["batch_size"])

    common = dict(
        batch_size=int(data_cfg["batch_size"]),
        collate_fn=numpy_collate,
        num_workers=int(data_cfg.get("num_workers", 0)),
        drop_last=False,
    )
    return DataLoader(train_dataset, **common), DataLoader(test_dataset, **common)
