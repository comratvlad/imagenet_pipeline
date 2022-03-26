import pydoc
from functools import partial
from typing import Mapping

import albumentations as A
import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from lib.data.dataset import FolderDataset
from lib.losses.weighted_sum_loss import WeightedSumLoss, LossSettings


class ConfigParser(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super(ConfigParser, self).__init__()
        self.config = config
        self.experiment = config.experiment
        self.task = config.task

        self.model = hydra.utils.instantiate(config.model)
        self.model_input_feature = config.model_input_feature

        self.post_processing = pydoc.locate(config.post_processing)

        self.loss = self._get_weighted_sum_loss(config.losses, config.device)

        self.metrics_dict = {name: pydoc.locate(value) for name, value in config.metrics.items()}

    def train_dataloader(self) -> DataLoader:
        train_datasets, dataset_weights = [], []
        for _, train_dataset_setting in self.config.train_data.items():
            transforms = pydoc.locate(self.config.transforms)
            if self.config.augmentations:
                transforms = partial(transforms,
                                     albumentations_compose=ConfigParser._make_albumentations_pipeline(self.config.augmentations))
            dataset = FolderDataset(train_dataset_setting.path, train_dataset_setting.info_path,
                                    features=[pydoc.locate(feature) for feature in self.config.sampled_features],
                                    transforms=transforms, filter_by=train_dataset_setting.filter_by)
            train_datasets.append(dataset)
            dataset_weights.append(train_dataset_setting.weight if 'weight' in train_dataset_setting else 1.)

        samples_weights = np.concatenate([[weight / sum(map(len, train_datasets))] * len(d)
                                         for weight, d in zip(dataset_weights, train_datasets)])
        num_samples = self.config.epoch_length * self.config.batch_size if self.config.epoch_length else len(samples_weights)

        return DataLoader(ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0],
                          batch_size=self.config.batch_size,
                          sampler=WeightedRandomSampler(Tensor(samples_weights), num_samples=num_samples),
                          num_workers=self.config.num_workers, pin_memory=True)

    def val_dataloader(self) -> Mapping[str, DataLoader]:
        dev_datasets = {}
        for name, dev_dataset_setting in self.config.dev_data.items():
            transforms = pydoc.locate(self.config.transforms)
            if self.config.augmentations:
                transforms = partial(transforms,
                                     albumentations_compose=ConfigParser._make_albumentations_pipeline(self.config.augmentations))
            dataset = FolderDataset(dev_dataset_setting.path, dev_dataset_setting.info_path,
                                               features=[pydoc.locate(feature) for feature in self.config.sampled_features],
                                               transforms=transforms, filter_by=dev_dataset_setting.filter_by)
            dev_datasets[name] = DataLoader(dataset,
                                            batch_size=self.config.batch_size,
                                            num_workers=self.config.num_workers)
        return CombinedLoader(dev_datasets)

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
