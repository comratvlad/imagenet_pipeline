from typing import Mapping

import torch


def logit_processing(logits: torch.Tensor) -> Mapping:
    return {'logits': logits,
            'probabilities': torch.softmax(logits, dim=1) if logits.size()[-1] != 1 else torch.sigmoid(logits)}
