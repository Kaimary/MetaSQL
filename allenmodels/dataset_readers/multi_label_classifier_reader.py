from collections import OrderedDict, defaultdict
import itertools
import json
import logging
import numpy as np
from typing import Dict, List, Optional


from allennlp.data import Tokenizer
from allennlp.data.instance import Instance
from allennlp.data.fields import MultiLabelField, TextField, ArrayField, Field, MetadataField
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from .enc_preproc import EncPreproc
# import dataset_utils.hashing as hashing

logger = logging.getLogger(__name__)

@DatasetReader.register("metadata_reader")
class MetadataDatasetReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Optional[Tokenizer] = None,
        keep_if_unparsable: bool = True,
        tables_file: str = "data/tables.json",
        dataset_path: str = "data/database",
        include_table_name_in_column=True,
        fix_issue_16_primary_keys=False,
        qq_max_dist=2,
        cc_max_dist=2,
        tt_max_dist=2,
        value_pred=True,
        use_longdb=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = tokenizer
        
        self.cls_token = self._tokenizer.tokenize("a")[0]
        self.eos_token = self._tokenizer.tokenize("a")[-1]
        self._keep_if_unparsable = keep_if_unparsable
        self.value_pred = value_pred
        params = open("params.txt")
        self._tables_file = params.readline().strip()
        self._dataset_path = params.readline().strip()
        
        # ratsql
        self.enc_preproc = EncPreproc(
            tables_file,
            dataset_path,
            include_table_name_in_column,
            fix_issue_16_primary_keys,
            qq_max_dist,
            cc_max_dist,
            tt_max_dist,
            use_longdb,
        )
    
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["enc"].token_indexers = self._token_indexers

    def _read(self, file_path):
        if not file_path.endswith('.json'):
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

        with open(file_path, "r") as data_file:
            json_obj = json.load(data_file)
            for ex in json_obj:
                if 'tags' in ex.keys():
                    tags = ex['tags'] if ex['tags'] else 'none'
                    ins = self.text_to_instance(
                        db_id = ex['db_id'],
                        utterance=ex['question'],
                        labels=str(ex['rating']) + ', ' + tags)
                        # labels=tags)
                else:
                    ins = self.text_to_instance(db_id = ex['db_id'], utterance=ex['question'])

                if ins is not None:
                    yield ins
        
    def text_to_instance(self, db_id: str, utterance: str, labels: str = None) -> Optional[Instance]:
        fields: Dict[str, Field] = {}

        tokenized_utterance = self._tokenizer.tokenize(utterance)
        desc = self.enc_preproc.get_desc(tokenized_utterance, db_id)
        entities, added_values, relation = self.extract_relation(desc)

        question_concated = [[x] for x in tokenized_utterance[1:-1]]
        schema_tokens_pre, schema_tokens_pre_mask = table_text_encoding(
            entities[len(added_values) + 1 :]
        )

        schema_size = len(entities)
        schema_tokens_pre = added_values + ["*"] + schema_tokens_pre

        schema_tokens = [
            [y for y in x if y.text not in ["_"]]
            for x in [self._tokenizer.tokenize(x)[1:-1] for x in schema_tokens_pre]
        ]

        fields["relation"] = ArrayField(relation, padding_value=-1, dtype=np.int32)
        
        enc_field_list = []
        offsets = []
        mask_list = (
            [False]
            + ([True] * len(question_concated))
            + [False]
            + ([True] * len(added_values))
            + [True]
            + schema_tokens_pre_mask
            + [False]
        )
        for mask, x in zip(
            mask_list,
            [[self.cls_token]]
            + question_concated
            + [[self.eos_token]]
            + schema_tokens
            + [[self.eos_token]],
        ):
            start_offset = len(enc_field_list)
            enc_field_list.extend(x)
            if mask:
                offsets.append([start_offset, len(enc_field_list) - 1])

        fields["lengths"] = ArrayField(
            np.array(
                [
                    [0, len(question_concated) - 1],
                    [len(question_concated), len(question_concated) + schema_size - 1],
                ]
            ),
            dtype=np.int32,
        )
        fields["offsets"] = ArrayField(
            np.array(offsets), padding_value=0, dtype=np.int32
        )
        fields["enc"] = TextField(enc_field_list, self._token_indexers)

        if labels:
            labels = [l.strip() for l in labels.split(',')]
            fields['labels'] = MultiLabelField(labels)

        return Instance(fields)

    def extract_relation(self, desc):
        def parse_col(col_list):
            col_type = col_list[0]
            col_name, table = "_".join(col_list[1:]).split("_<table-sep>_")
            return f'{table}.{col_name}:{col_type.replace("<type: ","")[:-1]}'

        question_concated = [x for x in desc["question"]]
        col_concated = [parse_col(x) for x in desc["columns"]]
        table_concated = ["_".join(x).lower() for x in desc["tables"]]
        enc = question_concated + col_concated + table_concated
        relation = self.enc_preproc.compute_relations(
            desc,
            len(enc),
            len(question_concated),
            len(col_concated),
            range(len(col_concated) + 1),
            range(len(table_concated) + 1),
        )
        unsorted_entities = col_concated + table_concated
        rel_dict = defaultdict(dict)
        # can do this with one loop
        for i, x in enumerate(list(range(len(question_concated))) + unsorted_entities):
            for j, y in enumerate(
                list(range(len(question_concated))) + unsorted_entities
            ):
                rel_dict[x][y] = relation[i, j]
        entities_sorted = sorted(list(enumerate(unsorted_entities)), key=lambda x: x[1])
        entities = [x[1] for x in entities_sorted]
        if self.value_pred:
            added_values = [
                "1",
                "2",
                "3",
                "4",
                "5",
                "yes",
                "no",
                "y",
                "t",
                "f",
                "m",
                "n",
                "null",
            ]
        else:
            added_values = ["value"]
        entities = added_values + entities
        new_enc = list(range(len(question_concated))) + entities
        new_relation = np.zeros([len(new_enc), len(new_enc)])
        for i, x in enumerate(new_enc):
            for j, y in enumerate(new_enc):
                if y in added_values or x in added_values:
                    continue
                new_relation[i][j] = rel_dict[x][y]
        return entities, added_values, new_relation

def table_text_encoding(entity_text_list):
    token_list = []
    mask_list = []
    for i, curr in enumerate(entity_text_list):
        if ":" in curr:  # col
            token_list.append(curr)
            if (i + 1) < len(entity_text_list) and ":" in entity_text_list[i + 1]:
                token_list.append(",")
            else:
                token_list.append(")\n")
            mask_list.extend([True, False])
        else:
            token_list.append(curr)
            token_list.append("(")
            mask_list.extend([True, False])

    return token_list, mask_list