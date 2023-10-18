from typing import List, Dict
from overrides import overrides

import numpy as np
import heapq
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.common.util import JsonDict, sanitize
from allennlp.predictors.predictor import Predictor
from configs.config import TOP_NUM
import logging
logger = logging.getLogger(__name__)

@Predictor.register("listwise-ranker")
class ListwiseRankingPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    # def predict(self, query: str = None, document: str = None) -> JsonDict:
    #     if not (query or document):
    #         logger.warn('Both query and document are empty. Skipping.')
    #         return {}
        
    #     return self.predict_json({'query': query, 'document': document})

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        json_output = {}
        outputs = self._model.forward_on_instance(instance)
        scores = outputs['logits']
        # pos = np.argmax(scores) #scores.index(max(scores))
        scores = scores.tolist()
        max_number = heapq.nlargest(TOP_NUM, scores) 
        max_index = []
        tmp = scores.index(max_number[0])
        for t in max_number:
            index = scores.index(t)
            max_index.append(str(index))
            scores[index] = 0
        for i in range(TOP_NUM - len(max_number)):
            max_index.append(str(tmp))

        pos = ' '.join(max_index)
        json_output['pos'] = pos
        return sanitize(json_output)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return str(outputs['pos']) + "\n"
