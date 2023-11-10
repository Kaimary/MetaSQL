import json
from typing import List, Union, Tuple
from overrides import overrides

from allennlp.data import DatasetReader
from allennlp.data.fields import ListField, MetadataField
from allennlp.data.instance import Instance
from allenmodels.dataset_readers.dataset_utils.utils import sql_format
from allenmodels.dataset_readers.ir_reader import IRDatasetReader

from utils.spider_utils.utils import find_relationship_tables, read_single_dataset_schema_from_database
from allenmodels.dataset_readers.dataset_utils.multi_grained_label_utils import query_to_scene_graph_labels, sql_to_phrases

@DatasetReader.register("listwise_pair_ranker_reader")
class ListwisePairRankingReader(IRDatasetReader):
    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as data_file:
            data = json.load(data_file)
            for ex in self.shard_iterable(data):
                # if len(json_obj) > 1034 and \
                #     ex['index'] in [209, 210, 333, 334, 585, 586, 601, 602, 2482, 2483, 2484, 2485, 3490, 3491]: continue
                # if ex['question'] != 'Show names of musicals and the number of actors who have appeared in the musicals.': continue
                # print(ex['index'])
                if 'labels' in ex.keys():
                    # print(ex)
                    ins = self.text_to_instance(
                        db_id=ex['db_id'],
                        query=ex['question'],
                        dialects=ex['candidates'],
                        labels=ex['labels'])
                else:
                    ins = self.text_to_instance(
                        db_id=ex['db_id'],
                        query=ex['question'],
                        dialects=ex['candidates'])

                if ins is not None:
                    yield ins
    
    def nl2phrases(self, nl):
        nl_entities, nl_triples = query_to_scene_graph_labels(nl)
        phrases = nl_entities + nl_triples # + ['' for _ in range(self._max_phrase_num)]

        # phrases_toks = []
        # for phrase in phrases:
        #     phrases_toks.append(self.tokenizer.tokenize(phrase))
        return phrases #[:self._max_phrase_num]

    def sql2phrases(self, sql, schema, db_id, relation_tables, kmaps, tables_file, database_path):
        object_entities, phrases = sql_to_phrases(sql, schema, db_id, relation_tables, kmaps, tables_file, database_path)
        clauses = object_entities + phrases if object_entities else phrases #+ ['' for _ in range(self._max_clause_num)]
        # clauses_toks = []
        # for clause in clauses:
        #     clauses_toks.append(self._tokenizer.tokenize(clause))
        return clauses #[:self._max_clause_num]
    
    def sep_pos_neg_clause(self, p_sql_phrases, n_sql_phrases):
        pos_mask = [0 for _ in range(len(n_sql_phrases))]
        neg_mask = [0 for _ in range(len(n_sql_phrases))]
        for idx, n_phrase in enumerate(n_sql_phrases):
            if n_phrase in p_sql_phrases:
                pos_mask[idx] = 1
            else:
                neg_mask[idx] = 1
        return pos_mask, neg_mask

    @overrides
    def text_to_instance(
        self,
        db_id: str,
        query: Union[str, Tuple], 
        dialects: List[str],
        labels: Union[str, float] = None
    ) -> Instance:  # type: ignore
        dialects = list(filter(None, dialects))
        if db_id not in self.visited_dbs:
            self.visited_dbs.append(db_id)
            self.schema = read_single_dataset_schema_from_database(db_id, self.database_path, self.tables_file)
            self.relation_tables = find_relationship_tables(self.schema)

        query_field = self._make_textfield(query)
        metadata = {'nl': query}
        # nl_phrases = self.nl2phrases(query)
        fields = {
            'nl': query_field,
            'metadata': MetadataField(metadata),
            # 'phrases': ListField([self._make_textfield(n_p) for n_p in nl_phrases]),
        }
        # print("\n")
        # print(f"NL: {query}")
        # print("Phrases", end =" ")
        # for n_p in nl_phrases: 
        #     print(f"{self._make_textfield(n_p).tokens}", end = " ")
        # print("\n")
        # print("-"*50)
        # all_clauses = []
        for idx, sql in enumerate(dialects):
            clauses = ['UNK'] # for _ in range(8)]
            try:
                clauses = self.sql2phrases(
                    sql, self.schema, db_id, self.relation_tables, self.kmaps, self.tables_file, self.database_path
                )
                if not clauses: clauses = ['UNK']
            except:
                pass
            # if any(any(sep in clause for sep in ['[DUPLICATE]', '[CONFLICT]', '[INACCURATE]', '[MISSING]']) for clause in clauses) and labels[idx] == 10.0:
            #     print(f"NL: {query}")
            #     print(f"SQL: {dialects[idx]}")
            # if any('UNK' in clause for clause in clauses):
            # if labels[idx] == 10.0:
            # print("SQL", end = ":")
            # print(sql)
            # print("Phrases", end =":")
            # for s_c in clauses: 
            #     # print(f"{self._make_textfield(s_c).tokens}", end = " ")
            #     print(f"{s_c}", end = ",")
            # print("\n")
            # print("*"*100)
            # fields[f'sql{idx}'] = self._make_textfield(sql_format(sql))
            fields[f'clauses{idx}'] = ListField([self._make_pair_textfield((query, s_c)) for s_c in clauses])
        #     all_clauses.append(clauses)
        # query_clauses_field = ListField([self._make_list_textfield([query] + clauses) for clauses in all_clauses])
        # fields['query_clauses'] = query_clauses_field 
        #     print(f"SQL {idx}: {self._make_textfield(sql_format(sql)).tokens}")
            
        # Format SQL
        ## 1. Remove table alias
        ## 2. Split aggregation with space
        ## 3. Capitalize sql keywords
        ## 4. Remove all underscores within columns
        dialects = [sql_format(dialect) for dialect in dialects]
        query_dialect_pairs_field = ListField([self._make_pair_textfield((query, o)) for o in dialects])
        fields['query_dialect_pairs'] = query_dialect_pairs_field 
        
        # if labels is not None:
        #     # Get the hardest (most similar) one as the negative sql
        #     g_sql, n_sql = "", ""
        #     sql_clauses, n_sql_clauses = [], []
        #     for iidx, label in zip(heapq.nlargest(10, range(len(labels)), key=labels.__getitem__), heapq.nlargest(10, labels)):
        #         if not g_sql and label == 10.0: 
        #             g_sql = dialects[iidx]
        #             # print(g_sql)
        #             sql_clauses = self.sql2phrases(
        #                 g_sql, self.schema, db_id, self.relation_tables, self.kmaps, self.tables_file, self.database_path
        #             )
        #             if not sql_clauses: sql_clauses = ['UNK']
        #         elif label < 10.0:
        #             n_sql = dialects[iidx]
        #             try:
        #                 sql_clauses2 = self.sql2phrases(
        #                     n_sql, self.schema, db_id, self.relation_tables, self.kmaps, self.tables_file, self.database_path
        #                 )
        #                 # if len(n_sql_clauses) == len(sql_clauses) and all(clause in sql_clauses for clause in n_sql_clauses): n_sql = ""
        #                 for clause in sql_clauses2:
        #                     if clause not in n_sql_clauses:
        #                         n_sql_clauses.append(clause)
        #             except:
        #                 # n_sql = ""
        #                 continue
        #         # print(f"n_sql_clauses:{len(n_sql_clauses)}")
            
        #     _, neg_mask_clause = self.sep_pos_neg_clause(p_sql_phrases=sql_clauses, n_sql_phrases=n_sql_clauses)

        #     indices = [i for i, n in enumerate(neg_mask_clause) if n == 1]
        #     count = 5 if len(indices) >= 5 else len(indices)
        #     pos = indices[count-1]
        #     neg_mask_clause = neg_mask_clause[:pos+1]
        #     n_sql_clauses = n_sql_clauses[:pos+1]
        #     # print(f"Gold SQL: {g_sql}")
        #     # # print(f"Negative SQL: {n_sql}")
        #     # print("Positive Phrases", end =" ")
        #     # for s_c in sql_clauses: 
        #     #     print(f"{self._make_textfield(s_c).tokens}", end = " ")
        #     # print("\n")
        #     # print("Negative Phrases", end =" ")
        #     # print(neg_mask_clause.count(1))
        #     # for s_c, mask in zip(n_sql_clauses, neg_mask_clause): 
        #     #     if mask == 1: print(f"{self._make_textfield(s_c).tokens}", end = " ")
        #     # print("\n")
        #     # print("*"*100)
            
        #     # fields['gold_sql'] = self._make_textfield(sql_format(g_sql))   
        #     fields['gold_clauses'] = ListField([self._make_textfield(s_c) for s_c in sql_clauses])
        #     # fields['neg_sql'] = self._make_textfield(sql_format(n_sql))
        #     fields['neg_clauses'] = ListField([self._make_textfield(n_s_c) for n_s_c in n_sql_clauses])
        #     # fields['pos_clause_mask'] = ArrayField(np.array(pos_mask_clause), padding_value=0, dtype=np.dtype(bool))
        #     fields['neg_clause_mask'] = ArrayField(np.array(neg_mask_clause), padding_value=0, dtype=np.dtype(bool))
        
        #     assert all(l >= 0 for l in labels)
        #     assert all((l == 0) for l in labels[len(dialects):])
        #     labels = labels[:len(dialects)]
        #     labels = list(map(float, filter(lambda x: not pd.isnull(x), labels)))            
        #     fields['labels'] = ArrayField(np.array(labels), padding_value=-1)
        
        fields = {k: v for (k, v) in fields.items() if v is not None}
        return Instance(fields)
