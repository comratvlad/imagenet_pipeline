from collections import defaultdict
from typing import Dict, List

import torch
import torchmetrics
from omegaconf import DictConfig
from torch import Tensor

from lib.config_parser import ConfigParser


class BaseClassifierPipeline(ConfigParser):
    def __init__(self, config: DictConfig):
        super(BaseClassifierPipeline, self).__init__(config)
        self.train_losses = defaultdict(float)
        self.train_batches_num = 0
        self.val_metrics = defaultdict(lambda: defaultdict(float))
        self.val_losses = defaultdict(lambda: defaultdict(float))
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x: Tensor):
        return self.model(x)

    def prepare_data(self) -> None:
        pass

    def training_step(self, batch: dict, _batch_idx: int) -> Tensor:
        inputs = batch[self.model_input_feature]
        outputs = self.post_processing(self.forward(inputs))
        weighted_sum, _components = self.loss(batch, outputs)
        self.train_losses = {k: val + self.train_losses[k] for k, val in _components.items()}
        self.log('train_loss', weighted_sum, on_step=False, on_epoch=True, prog_bar=False)
        self.train_batches_num += 1
        return weighted_sum

    def training_epoch_end(self, _trainingStepOutputs):
        self.train_losses = {k: i / self.train_batches_num for k, i in self.train_losses.items()}
        self.log('train_loss', self.train_losses, prog_bar=True)
        self.train_losses = defaultdict(float)
        self.train_batches_num = 0

    def validation_step(self, batch: dict, _batch_idx: int) -> Dict[str, Tensor]:
        for name, _batch in batch.items():
            inputs = _batch[self.model_input_feature]
            outputs = self.post_processing(self.forward(inputs))
            weighted_sum, _components = self.loss(_batch, outputs)
            self.accuracy(outputs['logits'], _batch['int_label'])
            for loss_name in _components:
                self.val_losses[name][loss_name] += _components[loss_name]
                self.log("val_loss_{}_{}".format(name, loss_name), _components[loss_name], prog_bar=True)
            self.log("val_acc", self.accuracy, prog_bar=True)
            return {'val_loss': weighted_sum}

    def validation_epoch_end(self, validationStepOutputs: List[Dict[str, Tensor]]):
        avgLoss = torch.stack([x['val_loss'] for x in validationStepOutputs]).mean()
        self.log('val_loss', avgLoss, prog_bar=True)
        self.val_metrics = defaultdict(lambda: defaultdict(float))
        self.val_losses = defaultdict(lambda: defaultdict(float))
