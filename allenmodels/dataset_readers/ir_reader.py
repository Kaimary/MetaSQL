import logging
from typing import Dict, Union, Tuple

from allennlp.data.fields import TextField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer

# from dataset_readers.reranker_reader import RelationExtractor
from allenmodels.dataset_readers.dataset_utils.schema import get_schema_from_json
from spider_utils.evaluation.evaluate1 import build_foreign_key_map_from_json
logger = logging.getLogger(__name__)


class IRDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        # super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        params = open("params.txt")
        self.tables_file = params.readline().strip()
        self.database_path = params.readline().strip()
        # self.tables_file = 'data/tables.json'
        # self.database_path = 'data/database'
        self.tables_json = get_schema_from_json(self.tables_file)
        self.kmaps = build_foreign_key_map_from_json(self.tables_file)
        # self.relation_extractor = RelationExtractor(self.database_path, self.tables_file)
        self.visited_dbs = []
        # self._max_phrase_num = 20
        # self._max_clause_num = 8

        # special_tokens_dict = {
        #     'additional_special_tokens': [
        #         "SELECT", "JOIN", "GROUP", "BY", "ORDER", "WHERE", "INTERSECT", "UNION", "EXCEPT", "FROM", "LIMIT", # 11
        #         "COUNT(", "SUM(", "AVG(", "MAX(", "MIN(", "*", # 6
        #         "[DUPLICATE]", "[FK]", "[MISSING]", "[CONFLICT]", "[MATCH]", # "[INACCURATE]", "[UNUSED]" # 7
        #         # "t0", "t1", "t2", "t3", "t4", "t5", "t6", # 7
        #         # "c0", "c1", "c2", "c3", "c4", "c5", "c6", # 7
        #         # "c0*", "c1*", "c2*", "c3*", "c4*", "c5*", "c6*" # 6
        #     ]
        # }
        # self.tokenizer.tokenizer.add_special_tokens(special_tokens_dict)
        # self.token_indexers['bert']._tokenizer.add_special_tokens(special_tokens_dict)

    def _make_textfield(self, text: Union[str, Tuple], remove_prefix=False):
        # if not text:
        #     return None
        
        if isinstance(text, tuple) and not isinstance(self.tokenizer, PretrainedTransformerTokenizer):
            text = ' '.join(text)
        
        tokens = self.tokenizer.tokenize(text)
        if remove_prefix: tokens = [t for t in tokens if t.text not in ['Ġ', 'table', 'column']]
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        return TextField(tokens, token_indexers=self.token_indexers)

    def _make_pair_textfield(self, text: Union[str, Tuple]):
        if not text:
            return None
        
        if isinstance(text, tuple):
            sentence_a, sentence_b = text
            tokens_a = self.tokenizer.tokenize(sentence_a)
            tokens_b = self.tokenizer.tokenize(sentence_b)
            # concat_tokens = self.tokenizer.add_special_tokens(tokens_a, tokens_b)
            concat_tokens = tokens_a + tokens_b
            if self.max_tokens:
                concat_tokens = concat_tokens[:self.max_tokens]
            
        return TextField(concat_tokens, token_indexers=self.token_indexers)
    
    def _make_list_textfield(self, texts):
        if not texts:
            return None
        
        t0 = self.tokenizer.tokenize(texts[0])
        for t1 in texts[1:]:
            t1 = self.tokenizer.tokenize(t1)
            # t0 = self.tokenizer.add_special_tokens(t0, t1)
            t0 = t0 + t1
        if self.max_tokens:
            t0 = t0[:self.max_tokens]
            
        return TextField(t0, token_indexers=self.token_indexers)