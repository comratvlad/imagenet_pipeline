import pydoc
from functools import partial
from typing import Union, Dict, Callable

import albumentations as A
import hydra
import omegaconf
import pytorch_lightning as pl
import timm.scheduler.scheduler
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.utils.data import DataLoader, ConcatDataset

from lib.data.dataset import FolderDataset
from lib.losses.weighted_sum_loss import WeightedSumLoss, LossSettings


class ConfigParser(pl.LightningModule):
    model: torch.nn.Module
    model_input_feature: str
    post_processing: Callable

    train_dataset: torch.utils.data.Dataset
    dev_datasets: Dict[str, torch.utils.data.Dataset]

    scheduler: Union[torch.optim.lr_scheduler._LRScheduler, timm.scheduler.scheduler.Scheduler, None]
    loss: WeightedSumLoss
    checkpoints: str

    def __init__(self, config: DictConfig):
        super(ConfigParser, self).__init__()
        self.config = config
        self.experiment = config.experiment
        self.task = config.task

        self.model = hydra.utils.instantiate(config.model)
        self.model_input_feature = config.model_input_feature

        self.post_processing = pydoc.locate(config.post_processing)

        self.loss = self._get_weighted_sum_loss(config.losses, config.device)
        self.n_epochs = config.n_epochs
        self.metrics_dict = {name: pydoc.locate(value) for name, value in config.metrics.items()}

    def train_dataloader(self) -> DataLoader:
        train_datasets = []
        for _, train_dataset_setting in self.config.train_data.items():
            transforms = pydoc.locate(self.config.transforms)
            if self.config.augmentations:
                transforms = partial(transforms,
                                     albumentations_compose=ConfigParser._make_albumentations_pipeline(self.config.augmentations))
            dataset = FolderDataset(train_dataset_setting.path, train_dataset_setting.info_path,
                                    features=[pydoc.locate(feature) for feature in self.config.sampled_features],
                                    transforms=transforms, filter_by=train_dataset_setting.filter_by)
            train_datasets.append(dataset)
        return DataLoader(ConcatDataset(train_datasets),
                          batch_size=self.config.batch_size,
                          pin_memory=True,
                          shuffle=True,
                          num_workers=self.config.num_workers)

    def val_dataloader(self) -> DataLoader:
        dev_datasets = {}
        for name, dev_dataset_setting in self.config.dev_data.items():
            transforms = pydoc.locate(self.config.transforms)
            if self.config.augmentations:
                transforms = partial(transforms,
                                     albumentations_compose=ConfigParser._make_albumentations_pipeline(self.config.augmentations))
            dev_datasets[name] = FolderDataset(dev_dataset_setting.path, dev_dataset_setting.info_path,
                                               features=[pydoc.locate(feature) for feature in self.config.sampled_features],
                                               transforms=transforms, filter_by=dev_dataset_setting.filter_by)
        return DataLoader(dev_datasets['imagenet1k'], batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def configure_optimizers(self) -> Optimizer:
        return hydra.utils.instantiate(self.config.optimizer, params=self.model.parameters())

    @staticmethod
    def _make_albumentations_pipeline(description):
        pipeline = []
        for item in description:
            if isinstance(item, dict) or omegaconf.OmegaConf.is_config(item):
                if len(item) != 1:
                    raise ValueError(f'String or dictionary with single key containing import string is expected '
                                     f'in every list item; got {item}')
                import_str, params = list(item.items())[0]
            else:
                import_str = item
                params = {}
            pipeline.append(pydoc.locate(import_str)(**params))
        return A.Compose(pipeline)

    @staticmethod
    def _get_weighted_sum_loss(losses_description, device):
        components = []
        try:
            for name, description in losses_description.items():
                loss_instance = hydra.utils.instantiate(description.callable)
                components.append(LossSettings(
                    name=name,
                    instance=loss_instance,
                    args=description.args,
                    weight=description.weight
                ))
        except KeyError as e:
            raise ValueError(f'Missing key "{e.args[0]}" in losses description') from e
        return WeightedSumLoss(*components, device=device)
