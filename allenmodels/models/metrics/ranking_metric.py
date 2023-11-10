from typing import List
import torch

from typing import Optional, List

import numpy as np
import torch

from allennlp.training.metrics.metric import Metric

class RankingMetric(Metric):
    def __init__(
        self,
        padding_value: int = -1
    ) -> None:
        self._padding_value = padding_value
        self.reset()
        
    def __call__(
            self,
            predictions: torch.LongTensor,
            gold_labels: torch.LongTensor,
            mask: torch.LongTensor = None
        ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of real-valued predictions of shape (batch_size, slate_length).
        gold_labels : ``torch.Tensor``, required.
            A tensor of real-valued labels of shape (batch_size, slate_length).
        """
        
        if mask is None:
            mask = torch.ones_like(gold_labels).bool()
        
        self._all_predictions.append(predictions.detach().cpu())
        self._all_gold_labels.append(gold_labels.detach().cpu()) 
        self._all_masks.append(mask.detach().cpu())
        
    @property
    def predictions(self):
        return torch.cat(self._all_predictions, dim=0)
    
    @property
    def gold_labels(self):
        return torch.cat(self._all_gold_labels, dim=0)
    
    @property
    def masks(self):
        return torch.cat(self._all_masks, dim=0)
        
    def get_metric(self, reset: bool = False):
        raise NotImplementedError()
    
    def reset(self):
        self._all_predictions = []
        self._all_gold_labels = []
        self._all_masks = []