import os, json, sqlite3, signal, threading, time
from typing import List, Union, Tuple
from overrides import overrides

from allennlp.data import DatasetReader
from allennlp.data.fields import ListField, MetadataField
from allennlp.data.instance import Instance
from allenmodels.dataset_readers.dataset_utils.utils import sql_format
from allenmodels.dataset_readers.ir_reader import IRDatasetReader

from utils.spider_utils.utils import find_relationship_tables, read_single_dataset_schema_from_database
from allenmodels.dataset_readers.dataset_utils.multi_grained_label_utils import sql_to_phrases

@DatasetReader.register("listwise_pair_ranker_reader")
class ListwisePairRankingReader(IRDatasetReader):
    def update_distinct_statistics(self, db_id, tables, key):
        cursor = sqlite3.connect(os.path.join(self.database_path, db_id, f'{db_id}.sqlite')).cursor()
        for t in tables:
            if not self.schema["primaries"][t]: continue
            # Get the primary keys (columns) of the related tables
            pk = self.schema["primaries"][t].pop()
            statement = f'SELECT COUNT(DISTINCT {t}.{pk}) FROM {key}'
            try:
                cursor.execute(statement)
                rows = [list(row) for row in cursor.fetchall()]
                self.t2distincts[f'{key}'].append((t, rows[0][0]))
            except:
                pass
        return

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as data_file:
            data = json.load(data_file)
            for ex in self.shard_iterable(data):
                if 'labels' in ex.keys():
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
                    # self.cnt += 1
                    yield ins
    
    def sql2phrases(self, sql, db_id):
        def interrupt(signum, frame):
            pass
        p_sql = rebuild_sql(db_id, self.database_path, sql.lower(), self.kmaps, self.tables_file, rebuild_col=False)
        # print(p_sql)
        if not isinstance(p_sql['from']['table_units'], dict):
            tables = sorted([t[1].strip('_') for t in p_sql['from']['table_units'] if isinstance(t[1], str)])
            key = ' JOIN '.join(tables)
            if tables and key not in self.t2distincts:
                signal.signal(signal.SIGINT, interrupt)
                mainthread = threading.Thread(target=self.update_distinct_statistics, args=[db_id, tables, key])
                mainthread.start()
                cnt = 0
                while mainthread.isAlive() and cnt < 100:
                    time.sleep(0.05)
                    cnt += 1
        object_entities, phrases = sql_to_phrases(sql, p_sql, self.schema, self.t2distincts, self.relation_tables)
        clauses = object_entities + phrases if object_entities else phrases
        return clauses
    
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
        # print(f'NL query: {query}')
        dialects = list(filter(None, dialects))
        if db_id not in self.visited_dbs:
            self.visited_dbs.append(db_id)
            self.schema = read_single_dataset_schema_from_database(db_id, self.database_path, self.tables_file)
            self.schema_dict[db_id] = self.schema
            self.relation_tables = find_relationship_tables(self.schema)
            self.relation_tables_dict[db_id] = self.relation_tables
        else: 
            self.schema = self.schema_dict[db_id]
            self.relation_tables = self.relation_tables_dict[db_id]
        query_field = self._make_textfield(query)
        metadata = {'nl': query}
        fields = {
            'nl': query_field,
            'metadata': MetadataField(metadata),
        }
        for idx, sql in enumerate(dialects):
            clauses = ['UNK'] # for _ in range(8)]
            try:
                clauses = self.sql2phrases(sql, db_id)
                if not clauses: clauses = ['UNK']
            except: pass
            fields[f'clauses{idx}'] = ListField([self._make_pair_textfield((query, s_c)) for s_c in clauses])
        # print(self.t2distincts)
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
        #             sql_clauses = self.sql2phrases(g_sql, db_id)
        #             if not sql_clauses: sql_clauses = ['UNK']
        #         elif label < 10.0:
        #             n_sql = dialects[iidx]
        #             try:
        #                 sql_clauses2 = self.sql2phrases(n_sql, db_id)
        #                 n_sql_clauses.extend([c for c in sql_clauses2 if c not in n_sql_clauses])
        #             except: 
        #                 continue
        #     # print(f"n_sql_clauses:{len(n_sql_clauses)}")
        #     # for sql in sql_clauses:
        #     #     print(sql)
        #     # print("="*100)
        #     _, neg_mask_clause = self.sep_pos_neg_clause(p_sql_phrases=sql_clauses, n_sql_phrases=n_sql_clauses)

        #     indices = [i for i, n in enumerate(neg_mask_clause) if n == 1]
        #     if not indices: return None
        #     count = 5 if len(indices) >= 5 else len(indices)
        #     pos = indices[count-1]
        #     neg_mask_clause = neg_mask_clause[:pos+1]
        #     n_sql_clauses = n_sql_clauses[:pos+1]
            
        #     # for nsql, mask in zip(n_sql_clauses, neg_mask_clause):
        #     #     if mask == 1: print(nsql)
        #     # print("="*100)
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