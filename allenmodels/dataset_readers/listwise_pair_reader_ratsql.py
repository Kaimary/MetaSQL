# -*- coding: utf-8 -*-
# @Time    : 2022/6/29 16:40
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : reranker_reader.py
# @Software: PyCharm
import copy
import json
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
import pickle
import os.path
import logging
import numpy as np
from tqdm import tqdm
from overrides import overrides
from allennlp.common import Params
from collections import defaultdict
from typing import Dict, Union, List

from allennlp.data.token_indexers import TokenIndexer
from allennlp.data import DatasetReader, Field, Instance, Tokenizer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.fields import TextField, MetadataField, ArrayField, ListField

from allenmodels.dataset_readers.dataset_utils.query_to_toks import is_number, to_number
from allenmodels.dataset_readers.dataset_utils.schema import get_schema_from_json, RESERVED_WORD
from allenmodels.dataset_readers.dataset_utils.utils import sql_format
from allenmodels.dataset_readers.ir_reader import IRDatasetReader

logger = logging.getLogger(__name__)
UNARY = {
    "min", "count", "max", "avg", "sum",
    "Subquery", "distinct", "literal"
}
AGGS = {
    "min", "count", "max", "avg", "sum",
}
FRONT_BINARY = {
    "Limit", "Orderby_desc", "Orderby_asc"
}


def _count_ex(ex: Dict):
    num = 0
    for key in ex.keys():
        num += key.startswith('level_')
    return num


@DatasetReader.register("listwise_pair_ranker_reader")
class RedDatasetReader(DatasetReader):
    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None,
            token_tokenizer: Tokenizer = None, # Error load pretrained tokenizer
            keep_if_unparsable: bool = True,
            dataset_path: str = "",
            database_path: str = "data/database",
            table_file: str = "data/tables.json",
            max_len=127,
            level_low=-1,
            level_high=10,
            **kwargs,
    ):
        super().__init__(**kwargs,)
        params = open("params.txt")
        tables_file = params.readline().strip()
        self.database_path = params.readline().strip()
        self.cached_data = defaultdict(set)
        self.tables_json = get_schema_from_json(tables_file)
        self.relation_extractor = RelationExtractor(self.database_path, tables_file)

        self._indexers = token_indexers
        self._tokenizer = token_tokenizer
        assert isinstance(self._tokenizer, PretrainedTransformerTokenizer), \
            "PretrainedTransformerTokenizer needed"
        special_tokens_dict = {
            'additional_special_tokens': [
                "SELECT", "JOIN", "GROUP", "BY", "ORDER", "WHERE", "INTERSECT", "UNION", "EXCEPT", "FROM", "LIMIT", # 11
                "COUNT(", "SUM(", "AVG(", "MAX(", "MIN(", "*", # 6
            ]
        }
        self._tokenizer.tokenizer.add_special_tokens(special_tokens_dict)
        self._indexers['bert']._tokenizer.add_special_tokens(special_tokens_dict)

        special_token = self._tokenizer.tokenize(self._tokenizer.tokenizer.sep_token)
        self.cls_token = special_token[0] #self._tokenizer.tokenizer.cls_token
        self.sep_token = special_token[1] #self._tokenizer.tokenizer.sep_token
        self.eos_token = special_token[2] #self._tokenizer.tokenizer.eos_token
        
        self._keep_if_unparsable = keep_if_unparsable
        self._dataset_path = dataset_path
        self.nl_max_len = max_len
        self.max_len = max_len * 2
        self._level_range = {i for i in range(level_low, level_high)}

        self.statistic = defaultdict(list)

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as data_file:
            json_obj = json.load(data_file)
            for ex in json_obj:
                if 'labels' in ex.keys():
                    candidates = []
                    for candidate in ex['candidates']:
                        candidate = sql_format(candidate)
                        one_candidate = [t for t in candidate.split()]
                        candidates.append(one_candidate)
                    ins = self.text_to_instance(
                        nl=ex['question'],
                        db_id=ex['db_id'],
                        candidates=candidates,
                        gold=ex['labels'])
                else:
                    ins = self.text_to_instance(
                        nl=ex['question'],
                        db_id=ex['db_id'],
                        candidates=candidates)

                if ins is not None:
                    yield ins

    def _ins_filter(self, instance: Instance):
        metadata = instance.fields['metadata']
        assert isinstance(metadata, MetadataField)
        return metadata['level_num'] in self._level_range


    def create_instance(self, ex) -> List[Instance]:
        ins_list = []
        nl = ex['nl']
        db_id = ex['db_id']
        for i in range(_count_ex(ex)):
            level = ex[f'level_{i}']
            gold = level['gold']
            candidates = [tree_reduce(tree) for tree in level['clause']]
            ins_list.append(self.text_to_instance(
                nl, db_id, candidates, gold, i
            ))
        return ins_list

    def text_to_instance(
            self,
            nl: str,
            db_id: str,
            candidates: List[List[str]],
            gold: List[bool] = None,
            level_num: int = -1
    ) -> Union[Instance, None]:
        if gold is not None:
            assert len(gold) == len(candidates)
            # gold = gold_expand(candidates, gold)
        metadata = {
            'db_id': db_id,
            'nl': nl,
            'level_num': level_num,
        }
        nl_tokens = self._tokenizer.tokenize(nl)[:self.nl_max_len]
        if nl_tokens[0] != self.cls_token:
            nl_tokens.insert(0, self.cls_token)
        elif nl_tokens[-1] != self.sep_token:
            nl_tokens[-1] = self.sep_token
        self.statistic['nl'].append(len(nl_tokens))
        nl_tokens_text = [tok.text.replace('Ġ', '') for tok in nl_tokens]

        candidate_fields = []
        candidates = candidates_sqlization(candidates, self.tables_json[db_id])
        max_candidate_len = max([len(candidate) for candidate in candidates]) + 1
        invert_index = []
        added_candidate = []
        reduced_gold = []
        candidates_offsets_list = []
        relations_list = []

        for i, candidate in enumerate(candidates):  # TODO: This candidate should be more natural
            # for i, tok in enumerate(candidate):
            #     candidate[i] = self.tables_json[db_id].get(tok, tok)
            candidate_str = ' '.join(candidate)
            # if candidate_str in added_candidate:
            #     invert_index.append(added_candidate.index(candidate_str))
            #     continue
            # else:
            invert_index.append(len(added_candidate))
            added_candidate.append(candidate_str)

            if gold:
                reduced_gold.append(gold[i])

            candidate_toks, candidate_masks = self.candidate_tokenize(candidate)

            # candidate_toks = self._tokenizer.tokenize(candidate_str)[1:self.max_len]
            if candidate_toks[-1] != self.eos_token:
                candidate_toks.append(self.eos_token)
                candidate_masks.append(True)
            self.statistic['candidate'].append(len(candidate_toks))
            candidate_fields.append(
                TextField(nl_tokens + candidate_toks, self._indexers)
            )

            candidates_offsets_list.append(
                self.cal_offsets(
                    nl_tokens, candidate_masks, len(nl_tokens) + max_candidate_len
                )
            )
            relations_list.append(
                self.extract_relations(
                    db_id, nl_tokens_text, candidate, len(nl_tokens) + max_candidate_len
                )
            )
        metadata['candidates'] = copy.deepcopy(candidates)

        fields: Dict[str, Field] = {
            "metadata": MetadataField(metadata),
            "candidates": ListField(candidate_fields),
            "invert_index": ArrayField(
                np.array(invert_index),
                padding_value=-1, dtype=np.dtype(np.int32)
            ),
            "offsets": ArrayField(
                np.array(candidates_offsets_list),
                padding_value=2048, dtype=np.dtype(np.int32)
            ),
            "relations": ArrayField(
                np.array(relations_list),
                padding_value=-1, dtype=np.dtype(np.int32)
            )
        }
        if gold:
            fields['gold'] = ArrayField(
                np.array(reduced_gold), padding_value=-1, dtype=np.dtype(np.int32)
            )
        return Instance(fields)

    def candidate_tokenize(self, candidate):
        tokens = []
        masks = []
        for token in candidate:
            tokenized = self._tokenizer.tokenize(token)[1:-1]
            if not tokenized:
                continue
            tokens += tokenized
            masks += [True] + [False] * (len(tokenized) - 1)
        assert len(tokens) == len(masks)
        tokens.append(self.eos_token)
        masks.append(True)
        return tokens[:self.max_len - 1], masks[:self.max_len - 1]

    @staticmethod
    def cal_offsets(nl_tokens, candidate_mask, padding_len=0):
        offsets = []
        nl_len = len(nl_tokens)
        for i in range(nl_len):
            offsets.append([i, i])
        s_idx = 0
        for i, m in enumerate(candidate_mask):
            if m:
                if s_idx:
                    offsets.append([s_idx, nl_len + i - 1])
                s_idx = nl_len + i
        offsets.append([s_idx, s_idx])
        if padding_len:
            for i in range(padding_len - len(offsets)):
                offsets.append([2048, 2048])
        return offsets

    def apply_token_indexers(self, instance: Instance) -> None:
        candidates = instance.fields["candidates"]
        assert isinstance(candidates, ListField)
        for enc in candidates:
            if isinstance(enc, TextField):
                enc.token_indexers = self._indexers
        return

    def extract_relations(self, db_id, nl_toks, candidate_toks, length=-1):
        real_relation = self.relation_extractor.extract_relation(
            db_id, [tok.lower() for tok in nl_toks], candidate_toks
        )
        if length > 0:
            relation = np.ones([length, length], dtype=int) * -1
            real_size = real_relation.shape[0]
            relation[:real_size, :real_size] = real_relation
            return relation
        else:
            return real_relation


class RelationExtractor:
    def __init__(self, database_path: str, tables_file: str):
        self.rels = [
            'default',
            'tab_tab_default',
            'col_col_default',
            'query_col_default',
            'col_query_default',
            'query_tab_default',
            'tab_query_default',
            'col_tab_default',
            'tab_col_default',
            'query_col_exact_match',
            'col_query_exact_match',
            'query_col_partial_match',
            'col_query_partial_match',
            'query_tab_exact_match',
            'tab_query_exact_match',
            'query_tab_partial_match',
            'tab_query_partial_match',
            'query_val_exact_match',
            'val_query_exact_match',
            'query_val_partial_match',
            'val_query_partial_match',
            'query_val_num_match',
            'val_query_num_match',
            'col_tab_in',
            'tab_col_in',
            'val_col_exact_match',
            'col_val_exact_match',
            'val_col_partial_match',
            'col_val_partial_match',
            'val_col_num_match',
            'col_val_num_match',
            'val_col_default',
            'col_val_default',
            'col_tab_primary_key',
            'tab_col_primary_key',
            'col_tab_foreign_key',
            'tab_col_foreign_key',
            'col_col_foreign_key_forward',
            'col_col_foreign_key_backward',
            'col_col_same_table',
            'tab_tab_foreign_key_forward',
            'tab_tab_foreign_key_backward',
            'tab_tab_foreign_key_both',
            'CLS_forward',
            'CLS_backward',
        ]
        self.rel_to_id = {}
        for i, rel in enumerate(self.rels):
            self.rel_to_id[rel] = i
        self.cached_invert_index = {}
        self.database_schema = load_schema(tables_file)
        self.database_path = database_path

    def parse_db(self, db_id):
        if db_id in self.cached_invert_index:
            return self.cached_invert_index[db_id]
        sqlite_path = Path(self.database_path) / db_id / f"{db_id}.sqlite"
        source: sqlite3.Connection
        col_vals = {}

        # Query db values
        with sqlite3.connect(sqlite_path, check_same_thread=False) as source:
            # Connection
            connection = sqlite3.connect(":memory:", check_same_thread=False)
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            source.backup(connection)
            # Query cols
            for col in self.database_schema[db_id]['cols']:
                tab_col, col_type = col.split(':')
                tab, col = tab_col.split('.')
                col_vals[tab_col] = set()
                # Query
                if col_type == 'number':
                    sql = f'SELECT min(`{col}`), max(`{col}`) FROM {tab} WHERE `{col}`'
                else:
                    sql = f'SELECT distinct `{col}` FROM {tab} LIMIT 5000'
                try:
                    cursor.execute(sql)
                    values = cursor.fetchall()
                    # Add values
                    if col_type == 'number':
                        try:
                            col_vals[tab_col].add((int(values[0][0]), int(values[0][1]) + 1))
                        except (TypeError, ValueError):
                            continue
                    else:
                        is_number_type = True
                        for val in values:
                            if is_number_type and not is_number(val[0]):
                                is_number_type = False
                            col_vals[tab_col].add(str(val[0]).lower())
                        # Fix number type
                        if is_number_type and col_vals[tab_col]:
                            if 'none' in col_vals[tab_col]:
                                col_vals[tab_col].remove('none')
                            if 'NULL' in col_vals[tab_col]:
                                col_vals[tab_col].remove('NULL')
                            if len(col_vals[tab_col]) < 1:
                                continue
                            vals = [float(val) for val in col_vals[tab_col]]
                            val = (int(min(vals)), int(max(vals)) + 1)
                            if None in val:
                                val = (1, -1)
                            col_vals[tab_col].clear()
                            col_vals[tab_col].add(val)
                except sqlite3.OperationalError:
                    print(f"SQL execution err on {db_id}: {sql}")
        # Build index
        self.cached_invert_index[db_id] = self.build_index(db_id, col_vals)
        return self.cached_invert_index[db_id]

    def build_index(self, db_id, value_info) -> Dict:
        index = {
            'nl_schema': {
                'num_match': {},
                'exact_match': set(),
                'partial_match': set()
            },
            'schema_schema': {
                'pk': set(),
                'fk': set(),
                # 'in': set(),
            },
            'nl': {
                'values': set(),
                'schemas': set()
            }
        }
        schema_info = self.database_schema[db_id]
        # Handle schema info

        # nl.schemas
        for col in schema_info['cols']:
            tab_col = col.split(':')[0]
            tab, col = tab_col.split('.')
            index['nl']['schemas'].add(tab_col)
            index['nl']['schemas'].add(tab.replace('_', ' '))
            index['nl']['schemas'].update(tab.split('_'))
            index['nl']['schemas'].add(col.replace('_', ' '))
            index['nl']['schemas'].update(col.split('_'))

            # nl_schema.partial_match
            index['nl_schema']['exact_match'].add(tab_col)
            index['nl_schema']['exact_match'].add(tab.replace('_', ' '))
            index['nl_schema']['exact_match'].add(col.replace('_', ' '))
            index['nl_schema']['partial_match'].update(tab.split('_'))
            index['nl_schema']['partial_match'].update(col.split('_'))

        # schema_schema.pk
        for pk in schema_info['pks']:
            tab = pk.split('.')[0]
            index['schema_schema']['pk'].add((tab, pk))

        # schema_schema_fk
        for fk in schema_info['fks']:
            index['schema_schema']['fk'].add(fk)
            index['schema_schema']['fk'].add(fk[0])
            index['schema_schema']['fk'].add(
                (fk[0].split('.')[0], fk[1].split('.')[0])
            )

        # nl.values
        for tab_col, values in value_info.items():
            if not values:
                continue
            tab = tab_col.split('.')[0]
            val_example = values.pop()
            values.add(val_example)
            if isinstance(val_example, str):
                for val in values:
                    if not val:
                        continue
                    val_tokens = val.split() + val.split('-')
                    index['nl']['values'].add(val)
                    index['nl']['values'].update(val_tokens)
                    index['nl_schema']['exact_match'].add((val, tab))
                    index['nl_schema']['exact_match'].add((val, tab_col))
                    for token in val_tokens:
                        index['nl_schema']['partial_match'].add((token, tab))
                        index['nl_schema']['partial_match'].add((token, tab_col))
            elif isinstance(val_example, tuple):
                index['nl_schema']['num_match'][tab_col] = val_example

        return index

    def extract_relation(
            self, db_id: str,
            nl_toks: List[str],
            candidate_toks: List[str]
    ):
        toks = nl_toks + candidate_toks
        nl_2_gram = [''.join(nl_toks[i: i+2]) for i in range(len(nl_toks))]
        nl_2_gram_space = [' '.join(nl_toks[i: i+2]) for i in range(len(nl_toks))]
        tok_2_gram = nl_2_gram + candidate_toks
        tok_2_gram_space = nl_2_gram_space + candidate_toks
        nl_len = len(nl_toks)
        toks_len = len(toks)
        index = self.parse_db(db_id)
        relations = np.zeros([toks_len, toks_len], dtype=int)
        # Handle relations between NL query and Candidate
        for toks in [toks, tok_2_gram, tok_2_gram_space]:
            for i in range(nl_len):
                for j in range(nl_len, toks_len):
                    rel = ['query', '', 'default']
                    rel[1] = self.candidate_tok_type(toks[j], db_id)
                    if rel[1] == 'default':
                        relations[i][j] = self.rel_to_id['default']
                        relations[j][i] = self.rel_to_id['default']
                    elif rel[1] == 'col' or rel[1] == 'tab':
                        if rel[1] == 'col' and is_number(toks[i]):
                            num_bound = index['nl_schema']['num_match'].get(toks[j], (1, -1))
                            if None in num_bound:
                                num_bound = (1, -1)
                            if num_bound[0] <= to_number(toks[i]) <= num_bound[1]:
                                rel[1] = 'val'
                                rel[2] = 'num_match'
                        if toks[i] in index['nl_schema']['exact_match']:
                            rel[2] = 'exact_match'
                        elif (toks[i], toks[j]) in index['nl_schema']['exact_match']:
                            rel[1] = 'val'
                            rel[2] = 'exact_match'
                        elif toks[i] in index['nl_schema']['partial_match']:
                            rel[2] = 'partial_match'
                        elif (toks[i], toks[j]) in index['nl_schema']['partial_match']:
                            rel[1] = 'val'
                            rel[2] = 'partial_match'
                        relations[i][j] = self.rel_to_id['_'.join(rel)]
                        rel[0], rel[1] = rel[1], rel[0]
                        relations[i][j] = self.rel_to_id['_'.join(rel)]
                    # elif rel[1] == 'val':
                    #     if toks[i] in index['nl_schema']['exact_match'] or \
                    #             (toks[i], toks[j]) in index['nl_schema']['exact_match']:
                    #         rel[2] = 'exact_match'
                    #     elif toks[i] in index['nl_schema']['partial_match'] or \
                    #             (toks[i], toks[j]) in index['nl_schema']['partial_match']:
                    #         rel[2] = 'partial_match'
        # Handle relations between Candidate and Candidate
        for i in range(nl_len, toks_len):
            for j in range(nl_len, toks_len):
                if i == j:
                    continue
                tok_i_type = self.candidate_tok_type(toks[i], db_id)
                tok_j_type = self.candidate_tok_type(toks[j], db_id)
                rel = [tok_i_type, tok_j_type, 'default']
                if tok_i_type == 'default' or tok_j_type == 'default':
                    relations[i][j] = self.rel_to_id['default']
                    relations[j][i] = self.rel_to_id['default']
                # col-col/tab/val relation, vice versa
                elif rel[0] == 'col':
                    # col-col relation
                    if rel[1] == 'col':
                        # same table
                        if toks[i].split('.')[0] == toks[j].split('.')[0]:
                            rel[2] = 'same_table'
                            relations[i][j] = self.rel_to_id['_'.join(rel)]
                            relations[j][i] = self.rel_to_id['_'.join(rel)]
                        # foreign key forward
                        elif (toks[i], toks[j]) in index['schema_schema']['fk']:
                            rel[2] = 'foreign_key_forward'
                            relations[i][j] = self.rel_to_id['_'.join(rel)]
                            rel[2] = 'foreign_key_backward'
                            relations[j][i] = self.rel_to_id['_'.join(rel)]
                        # col col default
                        else:
                            relations[i][j] = self.rel_to_id['_'.join(rel)]
                            relations[j][i] = self.rel_to_id['_'.join(rel)]
                    # col-tab relation
                    elif rel[1] == 'tab':
                        # primary key
                        if (toks[j].split('|')[0], toks[i]) in index['schema_schema']['pk']:
                            rel[2] = 'primary_key'
                        # fk or in
                        elif toks[i].split('.')[0] == toks[j].split('|')[0]:
                            if toks[i] in index['schema_schema']['fk']:
                                rel[2] = 'foreign_key'
                            else:
                                rel[2] = 'in'
                        relations[i][j] = self.rel_to_id['_'.join(rel)]
                        rel[0], rel[1] = rel[1], rel[0]
                        relations[j][i] = self.rel_to_id['_'.join(rel)]
                    # col-val relation
                    elif rel[1] == 'val':
                        if is_number(toks[j]):
                            num_bound = index['nl_schema']['num_match'].get(toks[i], (1, -1))
                            if None in num_bound:
                                num_bound = (1, -1)
                            if num_bound[0] <= to_number(toks[j]) <= num_bound[1]:
                                rel[2] = 'num_match'
                        if (toks[j], toks[i]) in index['nl_schema']['exact_match']:
                            rel[2] = 'exact_match'
                        elif (toks[j], toks[i]) in index['nl_schema']['partial_match']:
                            rel[2] = 'partial_match'
                        relations[i][j] = self.rel_to_id['_'.join(rel)]
                        rel[0], rel[1] = rel[1], rel[0]
                        relations[j][i] = self.rel_to_id['_'.join(rel)]
                # tab-tab relation
                elif rel[0] == 'tab' and rel[1] == 'tab':
                    tab_i = toks[i].split('|')[0]
                    tab_j = toks[j].split('|')[0]
                    if (tab_i, tab_j) in index['schema_schema']['fk'] and \
                            (tab_j, tab_i) in index['schema_schema']['fk']:
                        rel[2] = 'foreign_key_both'
                        relations[i][j] = self.rel_to_id['_'.join(rel)]
                        relations[j][i] = self.rel_to_id['_'.join(rel)]
                    elif (tab_i, tab_j) in index['schema_schema']['fk']:
                        rel[2] = 'foreign_key_forward'
                        relations[i][j] = self.rel_to_id['_'.join(rel)]
                        rel[2] = 'foreign_key_backward'
                        rel[0], rel[1] = rel[1], rel[0]
                        relations[j][i] = self.rel_to_id['_'.join(rel)]
        # CLS token linking
        relations[0, :] = self.rel_to_id['CLS_forward']
        relations[:, 0] = self.rel_to_id['CLS_backward']
        return relations

    @lru_cache(maxsize=256)
    def candidate_tok_type(self, tok: str, db_id: str):
        index = self.parse_db(db_id)
        if tok in RESERVED_WORD:
            return 'default'
        elif '.' in tok and tok in index['nl']['schemas']:
            return 'col'
        elif '|' in tok and tok.split('|')[0] in index['nl']['schemas']:
            return 'tab'
        elif tok in index['nl']['values'] or is_number(tok):
            return 'val'
        return 'default'


def load_schema(tables_file: str):
    """
    Load schema from tables.json, to form the schema linking index
    :param tables_file: table file path
    :return: table schema info
    """
    res = {}
    with open(tables_file, 'r') as f:
        schema_json = json.load(f)
    for db in schema_json:
        db_schema = {
            'tabs': [], 'cols': [], 'pks': [], 'fks': [],
        }
        for tab in db['table_names_original'] + db['table_names']:
            db_schema['tabs'].append(tab.replace(' ', '_'))
        for col, col_type in zip(db['column_names_original'], db['column_types']):
            if col[0] < 0:
                continue
            tab = db_schema['tabs'][col[0]]
            db_schema['cols'].append(f"{tab}.{col[1]}:{col_type}".lower())
        for pk in db['primary_keys']:
            db_schema['pks'].append(db_schema['cols'][pk - 1].split(':')[0])
        for (fk, pk) in db['foreign_keys']:
            fk_pair = (
                db_schema['cols'][fk - 1].split(':')[0],
                db_schema['cols'][pk - 1].split(':')[0]
            )
            db_schema['fks'].append(fk_pair)
        res[db['db_id']] = db_schema
    return res


def tree_reduce(tree: List[str]):
    tree_size = len(tree)
    root = int(tree_size / 2)
    if tree_size == 1:
        return [tree[0]]
    elif tree[root] == 'keep':
        return tree_reduce(tree[:root])
    elif tree[root] in UNARY:
        return [tree[root]] + tree_reduce(tree[:root])
    elif tree[root] == 'Project':
        if 'Selection' in tree[root + 1:]:
            return ['Project'] + tree_reduce(tree[:root]) + tree_reduce(tree[root + 1:])
        else:
            return ['Project'] + tree_reduce(tree[:root]) + ['from'] + tree_reduce(tree[root + 1:])
    elif tree[root] == 'Selection':
        l_tree = tree_reduce(tree[:root])
        r_tree = tree_reduce(tree[root + 1:])
        if ''.join(l_tree).strip():
            return ['Selection'] + l_tree + ['from'] + r_tree
        else:
            return ['from'] + r_tree
    elif tree[root] == 'Groupby':
        return ['Groupby'] + tree_reduce(tree[:root]) + tree_reduce(tree[root + 1:])
    elif tree[root] in FRONT_BINARY:
        return [tree[root]] + tree_reduce(tree[:root]) + tree_reduce(tree[root + 1:])
    else:
        return tree_reduce(tree[:root]) + [tree[root]] + tree_reduce(tree[root + 1:])


def candidates_sqlization(candidates: List[List[str]], token_mapping: Dict[str, str]):
    res = []
    for ori_candidate in candidates:
        candidate = candidate_sqlization(ori_candidate)
        for i, token in enumerate(candidate):
            candidate[i] = token_mapping.get(token, token)
        candidate = [token for token in candidate if token.strip() != '']
        res.append(candidate)
    return res


def candidate_sqlization(candidate: List[str]) -> List[str]:
    if not ('from' in candidate and 'Product' in candidate):
        return candidate
    for i, token in enumerate(candidate):
        if token == 'eq' and '.' in candidate[i + 1] and '.' in candidate[i - 1]:
            candidate[i - 1] = ''
            candidate[i] = ''
            candidate[i + 1] = ''
        elif token == 'And' and candidate[i - 1] == '':
            candidate[i] = ''
        elif token == 'from':
            while i > 0:
                i -= 1
                if candidate[i] != '':
                    if candidate[i] == 'Selection':
                        candidate[i] = ''
                    break
    return candidate


# def tree_to_sql(tree: List):
#     tree_size = len(tree)
#     root = tree_size // 2
#
#     if tree_size == 1:
#         return tree[root]
#     elif tree[root] in AGGS:
#         return "".join([tree[root], "( ", tree_to_sql(tree[:root-1]), " )"])
#     elif tree[root] == "distinct":
#         return "DISTINCT " + tree_to_sql(tree[:root-1])
#     elif tree[root] == "literal":
#         return "\'" + str(tree_to_sql(tree[:root-1])) + "\'"
#     elif tree[root] == "Subquery":
#         return "".join(["(", tree_to_sql(tree[:root-1]), ")"])
#     elif tree.name == "Join_on":
#         tree = tree.children[0]
#         if tree.name == "eq":
#             first_table_name = tree.children[0].val.split(".")[0]
#             second_table_name = tree.children[1].val.split(".")[0]
#             return f"{first_table_name} JOIN {second_table_name} ON {tree.children[0].val} = {tree.children[1].val}"
#         else:
#             if len(tree.children) > 0:
#                 t_Res = ", ".join([child.val for child in tree.children])
#                 return t_Res
#             else:
#                 return tree.val
#     else:  # Predicate or Table or 'literal' or Agg
#         return irra_to_sql(tree.children[0])
#     else:
#     if tree.name in [
#         "eq",
#         "like",
#         "nin",
#         "lte",
#         "lt",
#         "neq",
#         "in",
#         "gte",
#         "gt",
#         "And",
#         "Or",
#         "except",
#         "union",
#         "intersect",
#         "Product",
#         "Val_list",
#     ]:
#         pren_t = tree.name in [
#             "eq",
#             "like",
#             "nin",
#             "lte",
#             "lt",
#             "neq",
#             "in",
#             "gte",
#             "gt",
#         ]
#         return (
#             pred_dict[tree.name]
#                 .upper()
#                 .join([irra_to_sql(child, pren_t) for child in tree.children])
#         )
#     elif tree.name == "Orderby_desc":
#         return (
#                 irra_to_sql(tree.children[1])
#                 + " ORDER BY "
#                 + irra_to_sql(tree.children[0])
#                 + " DESC"
#         )
#     elif tree.name == "Orderby_asc":
#         return (
#                 irra_to_sql(tree.children[1])
#                 + " ORDER BY "
#                 + irra_to_sql(tree.children[0])
#                 + " ASC"
#         )
#     elif tree.name == "Project":
#         return (
#                 "SELECT "
#                 + irra_to_sql(tree.children[0])
#                 + " FROM "
#                 + irra_to_sql(tree.children[1])
#         )
#     elif tree.name == "Join_on":
#         # tree
#         def table_name(x):
#             return x.val.split(".")[0]
#
#         table_tups = [
#             (table_name(child.children[0]), table_name(child.children[1]))
#             for child in tree.children
#         ]
#         res = table_tups[0][0]
#         seen_tables = set(res)
#         for (first, sec), child in zip(table_tups, tree.children):
#             tab = first if sec in seen_tables else sec
#             res += (
#                 f" JOIN {tab} ON {child.children[0].val} = {child.children[1].val}"
#             )
#             seen_tables.add(tab)
#
#         return res
#     elif tree.name == "Selection":
#         if len(tree.children) == 1:
#             return irra_to_sql(tree.children[0])
#         return (
#                 irra_to_sql(tree.children[1])
#                 + " WHERE "
#                 + irra_to_sql(tree.children[0])
#         )
#     else:  # 'Selection'/'Groupby'/'Limit'/Having'
#         return (
#                 irra_to_sql(tree.children[1])
#                 + else_dict[tree.name]
#                 + irra_to_sql(tree.children[0])
#         )


def gold_expand(candidates: List[List[str]], gold: List[bool]):
    gold_candidates = list()
    for i, g in enumerate(gold):
        if g:
            gold_candidates.append(' '.join(candidates[i]))
    for i, candidate in enumerate(candidates):
        if ' '.join(candidate) in gold_candidates:
            gold[i] = True
    return gold


def test_load_schema():
    relation_extractor = RelationExtractor('datasets/sparc/database', 'datasets/sparc/tables.json')
    for db_id in tqdm(relation_extractor.database_schema.keys()):
        relation_extractor.parse_db(db_id)


def reranker_dataset_reader():
    print("\n\n[OUT] for test_reranker_dataset_reader:")
    pre_training_config = {
        "tokens": {
            "type": "pretrained_transformer",
            "model_name": "Salesforce/grappa_large_jnt",
        },
    }
    dataset_path = "./experiments/sparc-220629083933/reranker_data/train.json"
    dataset_reader = {
        "type": "red_reader",
        "token_indexers": copy.deepcopy(pre_training_config),
        "token_tokenizers": copy.deepcopy(pre_training_config),
        "dataset_path": dataset_path
    }
    ds = DatasetReader.from_params(
        params=Params(dataset_reader),
    )
    s_time = time.time()
    count = 0
    len_list = []
    for ex in tqdm(ds.read(dataset_path)):
        nl_len = len(ex.fields['metadata'].metadata['nl'].split())
        for tf in ex.fields['candidates'].field_list:
            len_list.append(len(tf) - nl_len)
        count += 1
    print(f"Average NL_SQL pair length {sum(len_list) / len(len_list)}")
    print(f"Read time: {time.time() - s_time:.2f}s")
    print(f"Read num : {count}")

# if __name__ == '__main__':
#     # reranker_dataset_reader()
#     # test_load_schema()
#     relation_extractor = RelationExtractor('datasets/sparc/database')
#     relation_extractor.parse_db('pets_1')
