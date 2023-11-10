from typing import Optional, List

import numpy as np
import torch

from allennlp.training.metrics.metric import Metric

from .ranking_metric import RankingMetric


@Metric.register("mrr")
class MRR(RankingMetric):
    def get_metric(self, reset: bool = False):
        predictions = torch.cat(self._all_predictions, dim=0)
        labels = torch.cat(self._all_gold_labels, dim=0)
        masks = torch.cat(self._all_masks, dim=0)
        
        score = mrr(predictions, labels, masks).item()

        if reset:
            self.reset()
        return score
    
    
# https://stackoverflow.com/a/60202801/6766123
def first_nonzero(t):
    t = t.masked_fill(t != 0, 1)
    idx = torch.arange(t.size(-1), 0, -1).type_as(t)
    indices = torch.argmax(t * idx, 1, keepdim=True)
    return indices


def mrr(y_pred, y_true, mask):
    y_pred = y_pred.masked_fill(~mask, -1)
    y_true = y_true.ge(y_true.max(dim=-1, keepdim=True).values).float()

    _, idx = y_pred.sort(descending=True, dim=-1)
    ordered_truth = y_true.gather(1, idx)
    
    gold = torch.arange(y_true.size(-1)).unsqueeze(0).type_as(y_true)
    _mrr = (ordered_truth / (gold + 1)) * mask
    
    return _mrr.gather(1, first_nonzero(ordered_truth)).mean()
