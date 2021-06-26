import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from lib.pl_module import BaseClassifier


@hydra.main()
def main(config: DictConfig) -> None:
    module = BaseClassifier(config)
    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0,
                      max_epochs=module.n_epochs)
    trainer.fit(module)


if __name__ == '__main__':
    main()
