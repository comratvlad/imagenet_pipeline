import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from lib.base_classifier import BaseClassifierPipeline


@hydra.main(config_path='../data')
def main(config: DictConfig) -> None:
    module = BaseClassifierPipeline(config)
    wandb_logger = WandbLogger(name=config.task, project=config.experiment, log_model=False)
    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0,
                      max_epochs=module.config.n_epochs, logger=wandb_logger)
    trainer.fit(module)


if __name__ == '__main__':
    main()
