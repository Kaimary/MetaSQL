"""
Utility functions for reading the standardised text2sql datasets presented in
`"Improving Text to SQL Evaluation Methodology" <https://arxiv.org/abs/1806.09029>`_
"""
import re
import json
import os
import sqlite3
import random
import networkx as nx
from copy import deepcopy
from unidecode import unidecode
from collections import defaultdict
from typing import List, Dict, Optional, Any

from .evaluation.process_sql import get_tables_with_alias, parse_sql

COLUMN_TYPE_MAP = {
    'text': 'text',
    'varchar': 'text',
    'char': 'text',
    'varchar2': 'text',
    'character varchar': 'text',
    'integer': 'number',
    'int': 'number',
    'real': 'number',
    'numeric': 'number',
    'decimal': 'number',
    'float': 'number',
    'double': 'number',
    'number': 'number',
    'smallint': 'number',
    'smallint unsigned': 'number',
    'mediumint unsigned': 'number',
    'tinyint unsigned': 'number',
    'bigint': 'number',
    'datetime': 'time',
    'timestamp': 'time',
    'date': 'time',
    'year': 'time',
    'bool': 'others',
    'boolean': 'others',
    'bit': 'others',
    'blob': 'others'
}

class TableColumn:
    def __init__(self,
                 name: str,
                 text: str,
                 column_type: str,
                 is_primary_key: bool,
                 foreign_key: Optional[str]):
        self.name = name
        self.text = text
        self.column_type = column_type
        self.is_primary_key = is_primary_key
        self.foreign_key = foreign_key


class Table:
    def __init__(self,
                 name: str,
                 text: str,
                 columns: List[TableColumn]):
        self.name = name
        self.text = text
        self.columns = columns


def read_dataset_schema(schema_path: str) -> Dict[str, List[Table]]:
    schemas: Dict[str, Dict[str, Table]] = defaultdict(dict)
    dbs_json_blob = json.load(open(schema_path, "r"))
    for db in dbs_json_blob:
        db_id = db['db_id']

        column_id_to_table = {}
        column_id_to_column = {}

        for i, (column, text, column_type) in enumerate(
                zip(db['column_names_original'], db['column_names'], db['column_types'])):
            table_id, column_name = column
            _, column_text = text

            table_name = db['table_names_original'][table_id].lower()

            if table_name not in schemas[db_id]:
                table_text = db['table_names'][table_id].lower()
                schemas[db_id][table_name] = Table(table_name, table_text, [])

            if column_name == "*":
                continue

            is_primary_key = i in db['primary_keys']
            table_column = TableColumn(column_name.lower(), column_text, column_type, is_primary_key, None)
            schemas[db_id][table_name].columns.append(table_column)
            column_id_to_table[i] = table_name
            column_id_to_column[i] = table_column

        for (c1, c2) in db['foreign_keys']:
            foreign_key = column_id_to_table[c2] + ':' + column_id_to_column[c2].name
            column_id_to_column[c1].foreign_key = foreign_key

    return {**schemas}


def read_single_dataset_schema(schema_path: str, db_id: str) -> Dict[str, List[Table]]:
    schema: Dict[str, Table] = defaultdict(dict)
    table = {}
    table_dict = {}
    dbs_json_blob = json.load(open(schema_path, "r"))
    for db in dbs_json_blob:
        if db_id == db['db_id']:

            table_dict = db
            table_dict['primaries'] = []
            for pk in table_dict['primary_keys']:
                column = table_dict['column_names_original'][pk]
                table_name = table_dict['table_names_original'][column[0]].lower()
                column_name = column[1].lower()
                table_dict['primaries'].append(table_name + '.' + column_name)

            column_id_to_table = {}
            column_id_to_column = {}

            table['column_names_original'] = db['column_names_original']
            table['table_names_original'] = db['table_names_original']

            for i, (column, text, column_type) in enumerate(
                    zip(db['column_names_original'], db['column_names'], db['column_types'])):
                table_id, column_name = column
                _, column_text = text

                table_name = db['table_names_original'][table_id].lower()

                if table_name not in schema:
                    table_text = db['table_names'][table_id].lower()
                    schema[table_name] = Table(table_name, table_text, [])

                if column_name == "*":
                    continue

                is_primary_key = i in db['primary_keys']
                table_column = TableColumn(column_name.lower(), column_text, column_type, is_primary_key, None)
                schema[table_name].columns.append(table_column)
                column_id_to_table[i] = table_name
                column_id_to_column[i] = table_column

            for (c1, c2) in db['foreign_keys']:
                foreign_key = column_id_to_table[c2] + ':' + column_id_to_column[c2].name
                column_id_to_column[c1].foreign_key = foreign_key

    return {**schema}, table, table_dict


def read_dataset_values(db_id: str, dataset_path: str, tables: List[str]):
    db = os.path.join(dataset_path, db_id, db_id + ".sqlite")
    try:
        conn = sqlite3.connect(db)
    except Exception as e:
        raise Exception(f"Can't connect to SQL: {e} in path {db}")
    conn.text_factory = str
    cursor = conn.cursor()

    values = {}

    for table in tables:
        try:
            cursor.execute(f"SELECT * FROM {table.name} LIMIT 5000")
            values[table] = cursor.fetchall()
        except:
            conn.text_factory = lambda x: str(x, 'latin1')
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table.name} LIMIT 5000")
            values[table] = cursor.fetchall()

    return values

def read_table_column_values(db_id: str, dataset_path: str, table: str, column1: str, column2: str = None):
    db = os.path.join(dataset_path, db_id, db_id + ".sqlite")
    try:
        conn = sqlite3.connect(db)
    except Exception as e:
        raise Exception(f"Can't connect to SQL: {e} in path {db}")
    conn.text_factory = str
    cursor = conn.cursor()

    try:
        if column2:
            cursor.execute(f"SELECT {column1}, {column2} FROM {table} LIMIT 5000")
            return cursor.fetchall()
        else:
            cursor.execute(f"SELECT {column1} FROM {table} LIMIT 5000")
            return cursor.fetchall()
    except:
        conn.text_factory = lambda x: str(x, 'latin1')
        cursor = conn.cursor()
        try:
            if column2:
                cursor.execute(f"SELECT {column1}, {column2} FROM {table} LIMIT 5000")
                return cursor.fetchall()
            else:
                cursor.execute(f"SELECT {column1} FROM {table} LIMIT 5000")
                return cursor.fetchall()
        except:
            return []


def ent_key_to_name(key):
    parts = key.split(':')
    if parts[0] == 'table':
        return parts[1]
    elif parts[0] == 'column':
        _, _, table_name, column_name = parts
        return f'{table_name}@{column_name}'
    else:
        return parts[1]


def fix_number_value(ex):
    """
    There is something weird in the dataset files - the `query_toks_no_value` field anonymizes all values,
    which is good since the evaluator doesn't check for the values. But it also anonymizes numbers that
    should not be anonymized: e.g. LIMIT 3 becomes LIMIT 'value', while the evaluator fails if it is not a number.
    """

    def split_and_keep(s, sep):
        if not s: return ['']  # consistent with string.split()

        # Find replacement character that is not used in string
        # i.e. just use the highest available character plus one
        # Note: This fails if ord(max(s)) = 0x10FFFF (ValueError)
        p = chr(ord(max(s)) + 1)

        return s.replace(sep, p + sep + p).split(p)

    # input is tokenized in different ways... so first try to make splits equal
    query_toks = ex['query_toks']
    ex['query_toks'] = []
    for q in query_toks:
        ex['query_toks'] += split_and_keep(q, '.')

    i_val, i_no_val = 0, 0
    while i_val < len(ex['query_toks']) and i_no_val < len(ex['query_toks_no_value']):
        if ex['query_toks_no_value'][i_no_val] != 'value':
            i_val += 1
            i_no_val += 1
            continue

        i_val_end = i_val
        while i_val_end + 1 < len(ex['query_toks']) and \
                i_no_val + 1 < len(ex['query_toks_no_value']) and \
                ex['query_toks'][i_val_end + 1].lower() != ex['query_toks_no_value'][i_no_val + 1].lower():
            i_val_end += 1

        if i_val == i_val_end and ex['query_toks'][i_val] in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"] and \
                ex['query_toks'][i_val - 1].lower() == "limit":
            ex['query_toks_no_value'][i_no_val] = ex['query_toks'][i_val]
        if ex['query_toks'][i_val - 2].lower() == "count" and ex['query_toks'][i_val].lower() == "1":
            ex['query_toks_no_value'][i_no_val] = "*"
        i_val = i_val_end

        i_val += 1
        i_no_val += 1

    return ex


_schemas_cache = None


def fix_query_toks_no_value(query_toks: List[str]):
    fixed_toks = []
    i = 0
    while i < len(query_toks):
        tok = query_toks[i]
        if tok == 'value' or tok == "'value'":
            # TODO: value should alawys be between '/" (remove first if clause)
            new_tok = f'"{tok}"'
        elif tok in ['!', '<', '>'] and query_toks[i + 1] == '=':
            new_tok = tok + '='
            i += 1
        elif i + 1 < len(query_toks) and query_toks[i + 1] == '.':
            new_tok = ''.join(query_toks[i:i + 3])
            i += 2
        else:
            new_tok = tok
        fixed_toks.append(new_tok)
        i += 1

    return ' '.join(fixed_toks)


def disambiguate_items(db_id: str, query_toks: List[str], tables_file: str, allow_aliases: bool) -> List[str]:
    """
    we want the query tokens to be non-ambiguous - so we can change each column name to explicitly
    tell which table it belongs to

    parsed sql to sql clause is based on supermodel.gensql from syntaxsql
    """

    class Schema:
        """
        Simple schema which maps table&column to a unique identifier
        """

        def __init__(self, schema, table):
            self._schema = schema
            self._table = table
            self._idMap = self._map(self._schema, self._table)

        @property
        def schema(self):
            return self._schema

        @property
        def idMap(self):
            return self._idMap

        def _map(self, schema, table):
            column_names_original = table['column_names_original']
            table_names_original = table['table_names_original']
            # print 'column_names_original: ', column_names_original
            # print 'table_names_original: ', table_names_original
            for i, (tab_id, col) in enumerate(column_names_original):
                if tab_id == -1:
                    idMap = {'*': i}
                else:
                    key = table_names_original[tab_id].lower()
                    val = col.lower()
                    idMap[key + "." + val] = i

            for i, tab in enumerate(table_names_original):
                key = tab.lower()
                idMap[key] = i

            return idMap

    def get_schemas_from_json(fpath):
        global _schemas_cache

        if _schemas_cache is not None:
            return _schemas_cache

        with open(fpath) as f:
            data = json.load(f)
        db_names = [db['db_id'] for db in data]

        tables = {}
        schemas = {}
        for db in data:
            db_id = db['db_id']
            schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
            column_names_original = db['column_names_original']
            table_names_original = db['table_names_original']
            tables[db_id] = {'column_names_original': column_names_original,
                             'table_names_original': table_names_original}
            for i, tabn in enumerate(table_names_original):
                table = str(tabn.lower())
                cols = [str(col.lower()) for td, col in column_names_original if td == i]
                schema[table] = cols
            schemas[db_id] = schema

        _schemas_cache = schemas, db_names, tables
        return _schemas_cache

    schemas, _, tables = get_schemas_from_json(tables_file)
    schema = Schema(schemas[db_id], tables[db_id])

    fixed_toks = []
    i = 0
    while i < len(query_toks):
        tok = query_toks[i]
        if tok == 'value' or tok == "'value'":
            # TODO: value should alawys be between '/" (remove first if clause)
            new_tok = f'"{tok}"'
        elif tok in ['!', '<', '>'] and query_toks[i + 1] == '=':
            new_tok = tok + '='
            i += 1
        elif i + 1 < len(query_toks) and query_toks[i + 1] == '.':
            new_tok = ''.join(query_toks[i:i + 3])
            i += 2
        else:
            new_tok = tok
        fixed_toks.append(new_tok)
        i += 1

    toks = fixed_toks

    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql, mapped_entities = parse_sql(toks, 0, tables_with_alias, schema, mapped_entities_fn=lambda: [])

    for i, new_name in mapped_entities:
        curr_tok = toks[i]
        if '.' in curr_tok and allow_aliases:
            parts = curr_tok.split('.')
            assert (len(parts) == 2)
            toks[i] = parts[0] + '.' + new_name
        else:
            toks[i] = new_name

    if not allow_aliases:
        toks = [tok for tok in toks if tok not in ['as', 't1', 't2', 't3', 't4']]

    toks = [f'\'value\'' if tok == '"value"' else tok for tok in toks]

    return toks


def disambiguate_items2(
        query_toks: List[str],
        schema: Dict[str, Any],
        table: Dict[str, List],
        allow_aliases: bool
) -> List[str]:
    """
    we want the query tokens to be non-ambiguous - so we can change each column name to explicitly
    tell which table it belongs to

    parsed sql to sql clause is based on supermodel.gensql from syntaxsql
    """

    class Schema:
        """
        Simple schema which maps table&column to a unique identifier
        """

        def __init__(self, schema, table):
            self._schema = schema
            self._table = table
            self._idMap, self._nameMap = self._map(self._schema, self._table)

        @property
        def schema(self):
            return self._schema

        @property
        def idMap(self):
            return self._idMap

        @property
        def nameMap(self):
            return self._nameMap

        def _map(self, schema, table):
            column_names_original = table['column_names_original']
            table_names_original = table['table_names_original']
            # print 'column_names_original: ', column_names_original
            # print 'table_names_original: ', table_names_original
            for i, (tab_id, col) in enumerate(column_names_original):
                if tab_id == -1:
                    idMap = {'*': -999}
                    nameMap = {-999: '*'}
                else:
                    key = table_names_original[tab_id].lower()
                    val = col.lower()
                    idMap[key + "." + val] = i
                    nameMap[i] = key + "." + val

            for i, tab in enumerate(table_names_original):
                key = tab.lower()
                idMap[key] = i + 999
                nameMap[i + 999] = key

            return idMap, nameMap

    schema = Schema(schema, table)

    fixed_toks = []
    i = 0
    while i < len(query_toks):
        tok = query_toks[i]
        if tok == 'value' or tok == "'value'":
            # TODO: value should alawys be between '/" (remove first if clause)
            new_tok = f'"{tok}"'
        elif tok in ['!', '<', '>'] and query_toks[i + 1] == '=':
            new_tok = tok + '='
            i += 1
        elif i + 1 < len(query_toks) and query_toks[i + 1] == '.':
            new_tok = ''.join(query_toks[i:i + 3])
            i += 2
        else:
            new_tok = tok
        fixed_toks.append(new_tok)
        i += 1
    toks = fixed_toks

    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql, mapped_entities = parse_sql(toks, 0, tables_with_alias, schema, mapped_entities_fn=lambda: [])
    for i, new_name in mapped_entities:
        curr_tok = toks[i]
        if '.' in curr_tok and allow_aliases:
            parts = curr_tok.split('.')
            assert (len(parts) == 2)
            toks[i] = parts[0] + '.' + new_name
        else:
            toks[i] = new_name

    if not allow_aliases:
        toks = [tok for tok in toks if tok not in ['as', 't1', 't2', 't3', 't4']]

    toks = [f'\'value\'' if tok == '"value"' else tok for tok in toks]

    return toks, sql, schema


def normalize_string(string: str) -> str:
    """
    These are the transformation rules used to normalize cell in column names in Sempre.  See
    ``edu.stanford.nlp.sempre.tables.StringNormalizationUtils.characterNormalize`` and
    ``edu.stanford.nlp.sempre.tables.TableTypeSystem.canonicalizeName``.  We reproduce those
    rules here to normalize and canonicalize cells and columns in the same way so that we can
    match them against constants in logical forms appropriately.
    """
    # Normalization rules from Sempre
    # \u201A -> ,
    string = re.sub("‚", ",", string)
    string = re.sub("„", ",,", string)
    string = re.sub("[·・]", ".", string)
    string = re.sub("…", "...", string)
    string = re.sub("ˆ", "^", string)
    string = re.sub("˜", "~", string)
    string = re.sub("‹", "<", string)
    string = re.sub("›", ">", string)
    string = re.sub("[‘’´`]", "'", string)
    string = re.sub("[“”«»]", "\"", string)
    string = re.sub("[•†‡²³]", "", string)
    string = re.sub("[‐‑–—−]", "-", string)
    # Oddly, some unicode characters get converted to _ instead of being stripped.  Not really
    # sure how sempre decides what to do with these...  TODO(mattg): can we just get rid of the
    # need for this function somehow?  It's causing a whole lot of headaches.
    string = re.sub("[ðø′″€⁄ªΣ]", "_", string)
    # This is such a mess.  There isn't just a block of unicode that we can strip out, because
    # sometimes sempre just strips diacritics...  We'll try stripping out a few separate
    # blocks, skipping the ones that sempre skips...
    string = re.sub("[\\u0180-\\u0210]", "", string).strip()
    string = re.sub("[\\u0220-\\uFFFF]", "", string).strip()
    string = string.replace("\\n", "_")
    string = re.sub("\\s+", " ", string)
    # Canonicalization rules from Sempre.
    string = re.sub("[^\\w]", "_", string)
    string = re.sub("_+", "_", string)
    string = re.sub("_$", "", string)
    return unidecode(string.lower())


def get_all_schema(file_path='sqlgenv2/datasets/spider/tables.json'):
    """
    get all spider's schema
    @param file_path: schema file path
    @return: all schemas dict
    """
    res = defaultdict(dict)
    with open(file_path, 'r') as f:
        db_schema_raw = json.load(f)
    for db_schema in db_schema_raw:
        # change into dict with id key
        res[db_schema['db_id']] = {
            'column': db_schema['column_names_original'],
            'column_types': db_schema['column_types'],
            'foreign_keys': db_schema['foreign_keys'],
            'primary_keys': db_schema['primary_keys'],
            'tables': db_schema['table_names_original'],
            'excludes': db_schema['excludes'] if 'excludes' in db_schema.keys() else [],
            'primary_cols': db_schema['primary_cols'] if 'primary_cols' in db_schema.keys() else [],
            'relationship_tables': db_schema['relationship_tables'] if 'relationship_tables' in db_schema.keys() else []
        }
    return res


class DBSchema(object):
    def __init__(self, db_id: str, db_schemas: dict, db_path='datasets/spider/database/'):
        self.db_id = db_id
        self.tables: list = db_schemas[db_id]['tables']
        self.columns: list = db_schemas[db_id]['column']
        self.column_types: list = db_schemas[db_id]['column_types']
        self.foreign_keys: list = db_schemas[db_id]['foreign_keys']
        self.primary_keys: list = db_schemas[db_id]['primary_keys']
        self.primary_cols: list = db_schemas[db_id]['primary_cols']
        self.excluded_columns: list = db_schemas[db_id]['excludes'] if 'excludes' in db_schemas[db_id].keys() else []
        self.relationship_tables: list = db_schemas[db_id]['relationship_tables'] if 'relationship_tables' in \
                                                                                     db_schemas[db_id].keys() else []
        self.column_table = defaultdict(list)
        self.table_column_num = defaultdict(int)
        self.table_column = defaultdict(list)
        self.table_ncolumn = defaultdict(list)
        self.table_scolumn = defaultdict(list)
        self.column_value = defaultdict(list)
        self.table_column_value = defaultdict(list)
        self.table_primary_key: list = list()
        self.table_foreign_key: dict = dict()
        self.__stat_column_from()
        self.__stat_table_column()
        if db_path:
            self.__read_values(db_path)
        self.__stat_primary_key()
        self.__stat_primary_col()
        self.__stat_foreign_key()

    def __stat_column_from(self):
        """
        Get revert index from columns to table
        @return: None
        """
        for item in self.columns[1:]:
            self.column_table[item[1]].append(self.tables[item[0]])
        return

    def __stat_table_column(self):
        """
        stat every table's columns num
        @return: None
        """
        for k, column in enumerate(self.columns[1:]):
            self.table_column_num[self.tables[column[0]]] += 1
            self.table_column[self.tables[column[0]].lower()].append(column[1].lower())
            if self.column_types[k + 1] == "number":
                self.table_ncolumn[self.tables[column[0]].lower()].append(column[1].lower())
            elif self.column_types[k + 1] == "text" or self.column_types[k + 1] == "time":
                self.table_scolumn[self.tables[column[0]].lower()].append(column[1].lower())
        return

    def get_tables(self, n, repeated=False):
        """
        get n table name randomly
        @param n: table number
        @param repeated: can choose one table more than one times
        @return: a list of table name
        """
        res = list()
        tables = deepcopy(self.tables)
        for _ in range(n):
            if not tables:
                raise IndexError("No choice any more")
            # res.append(random.choice(tables))
            # res.append(random.choice(tables).lower())  # TODO: if need to be lower case
            res.append(random.choices(tables, weights=[self.table_column_num[table] for table in tables])[0])
            # TODO: if need weights
            if not repeated:  # a table only choose once
                tables.remove(res[-1])
        return res

    def get_columns_by_tables(self, n, tables: list, exclude_columns=None, repeated=False):
        """
        get columns by table name, some choice will be forbidden, and control the column can or not repeat
        @param n: number of columns
        @param tables: choose from tables
        @param exclude_columns: can not be choose, for extendability
        @param repeated: can choose one column more than one times
        @return: a list of column name
        """
        if exclude_columns is None:  # no exclude columns
            exclude_columns = []
        res = list()
        table_index = {self.tables.index(table) for table in tables}
        columns = [column_item[-1] for column_item in self.columns if column_item[0] in
                   table_index and column_item[-1] not in exclude_columns]  # columns' name which can be choose
        for _ in range(n):
            if not columns:
                raise IndexError("No choice any more")
            choice = random.choice(columns)
            res.append(choice)
            # res.append(choice.lower())  # TODO: if need to be lower case
            if not repeated:  # can not repeat
                columns.remove(choice)
        return res

    def get_columns_by_constrict_tables(self, tables, exclude_columns=None):
        """
        choose constrict tables from both table
        @param tables: from table, which must more than one
        @param exclude_columns: can not be choose
        @return: one column name satisfy both tables
        """
        if exclude_columns is None:  # no exclude columns
            exclude_columns = []
        table_index = self.tables.index(tables[0])
        columns = {column_item[-1] for column_item in self.columns if column_item[0] ==
                   table_index and column_item[-1] not in exclude_columns}  # columns' name which can be choose
        for table in tables[1:]:  # check rest tables
            table_index = self.tables.index(table)
            new_columns = {column_item[-1] for column_item in self.columns if column_item[0] ==
                           table_index and column_item[-1] not in exclude_columns}
            columns = columns.intersection(new_columns)  # get intersection
        return random.choice(list(columns))

    def get_pks_by_tables(self, n, tables, exclude_columns=None, repeated=False):
        """
        get pk columns by table name, some choice will be forbidden, and control the column can or not repeat
        @param n: number of columns
        @param tables: choose from tables
        @param exclude_columns: can not be choose
        @param repeated: can choose one column more than one times
        @return: a list of column name
        """
        if exclude_columns is None:  # no exclude columns
            exclude_columns = []
        res = list()
        table_index = {self.tables.index(table) for table in tables}
        columns = [self.columns[pk_index][-1] for pk_index in
                   self.primary_keys if self.columns[pk_index][0] in table_index and
                   self.columns[pk_index][0] not in exclude_columns]  # columns' name which can be choose
        for _ in range(n):
            if not columns:
                raise IndexError("No choice any more")
            choice = random.choice(columns)
            res.append(choice)
            res.append(choice.lower())  # if need to be lower case
            if not repeated:  # can not repeat
                columns.remove(choice)
        return res

    def __read_values(self, db_path):
        """
        read values from databases
        @param db_path: database dir
        @return: None
        """
        db = os.path.join(db_path, self.db_id, self.db_id + ".sqlite")  # get path
        try:
            conn = sqlite3.connect(db)  # connect to db
        except Exception as e:
            raise Exception(f"Can't connect to SQL: {e} in path {db}")
        conn.text_factory = str
        cursor = conn.cursor()
        values = {}
        for table in self.tables:  # execute sql in every table
            try:
                cursor.execute(f"SELECT * FROM {table} LIMIT 50")
                values[table] = cursor.fetchall()
            except Exception as e:
                print(e)
                conn.text_factory = lambda x: str(x, 'latin1')
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table} LIMIT 5050")
                values[table] = cursor.fetchall()
        # append into column_value
        for table, rows in values.items():
            for row in rows:
                for column, data in zip(self.table_column[table.lower()], row):
                    # Skip None value
                    if data is None or data == "":
                        continue
                    self.column_value[column].append(data)
                    self.table_column_value[f'{table.lower()}.{column.lower()}'].append(data)
        return None

    def __stat_primary_key(self):
        for pk_col_index in self.primary_keys:
            column = self.columns[pk_col_index]
            table = self.tables[column[0]]
            self.table_primary_key.append(f"{table.lower()}.{column[1].lower()}")

    def __stat_primary_col(self):
        """
        get the col which can be considered as primary col, include the pk column
        such as xxx_name, xxx_code, xxx_number, xxx_id
        @return: None
        """
        self.primary_cols = set(list(map(lambda x: x.lower(), self.primary_cols)))

        if not self.primary_cols:
            for table in self.table_column:
                for column in self.table_column[table]:
                    for keyword in ['code', 'name', 'number', 'id']:
                        table_column_lower = f'{table.lower()}.{column.lower()}'
                        if keyword in table_column_lower:
                            self.primary_cols.add(table_column_lower)
        self.primary_cols.update(set(self.table_primary_key))

    def __stat_foreign_key(self):
        for fk_col_index_pair in self.foreign_keys:
            # get fk column name
            fk_col_index = fk_col_index_pair[0]
            fk_column = self.columns[fk_col_index]
            fk_table = self.tables[fk_column[0]]
            fk = f"{fk_table.lower()}.{fk_column[1].lower()}"
            # get pk column name
            pk_col_index = fk_col_index_pair[1]
            pk_column = self.columns[pk_col_index]
            pk_table = self.tables[pk_column[0]]
            pk = f"{pk_table.lower()}.{pk_column[1].lower()}"
            self.table_foreign_key[fk] = pk
        return


def remove_conds(c):
        if ' ON ' not in c and ' on ' not in c: return c

        toks = []
        join = -1
        sql_toks = c.split()
        for i, t in enumerate(sql_toks):
            if t in ['ON', 'on']:
                join = 3
                if i + 4 < len(sql_toks) and sql_toks[i + 4] in ['and', 'AND', 'or', 'OR']: join = 7
                continue
            elif join > 0: 
                join -= 1
            else: toks.append(t)
        
        return ' '.join(toks)


def read_single_dataset_schema_from_database(db_id: str, dataset_path: str, schema_path: str) -> Dict[str, List[Table]]:
    table_meta = defaultdict(dict)

    # Examine ``silver`` schema from schema file
    # It is silver since we found that in some cases, the definition in the schema file does not match the database
    dbs_json_blob = json.load(open(schema_path, "r"))
    table_meta.update([db for db in dbs_json_blob if db_id == db['db_id']][0])
    # Add additonal keys in the table_meta
    # Preprocessing the schema info defined by Spider author
    table_meta['primaries'] = defaultdict(set)
    for pk in table_meta['primary_keys']:
        # Format: [table_id, column_name]
        column = table_meta['column_names_original'][pk]
        table_name = table_meta['table_names_original'][column[0]].lower()
        table_meta['primaries'][table_name].add(column[1].lower())
    for column, text, type in zip(table_meta['column_names_original'][1:], table_meta['column_names'][1:], table_meta['column_types'][1:]):
        table_id, column_name = column
        _, column_text = text
        key = f"{table_meta['table_names_original'][table_id].lower()}.{column_name.lower()}"
        table_meta['column_names_mapping'][key] = column_text.lower()
        table_meta['column_names_types_mapping'][key] = type
    # Read the foreign keys from the schema file and use as a backup
    table_meta['foreigns_bak'], table_meta['foreign_key_columns_bak'] = [], []
    for (c1, c2) in table_meta['foreign_keys']:
        column = table_meta['column_names_original'][c1]
        table = table_meta['table_names_original'][column[0]]
        local_column = column[1]
        column2 = table_meta['column_names_original'][c2]
        foreign_table = table_meta['table_names_original'][column2[0]]
        foreign_column = column2[1]
        table_meta['foreign_key_columns_bak'].append(f'{table.lower()}.{local_column.lower()}')
        table_meta['foreigns_bak'].append(f'{table.lower()}-{foreign_table.lower()}')
        table_meta['foreign_column_pairs_bak'][f'{table.lower()}.{local_column.lower()}'] = f'{foreign_table.lower()}.{foreign_column.lower()}'
        table_meta['foreign_column_pairs_bak'][f'{foreign_table.lower()}.{foreign_column.lower()}'] = f'{table.lower()}.{local_column.lower()}'

    # Re-examine ``gold`` schema form SQLite database
    # 1. Re-check primary keys
    # 2. Get column types
    # 3. Get foreign keys
    try:
        db_path = os.path.join(dataset_path, db_id, db_id + ".sqlite")
        conn = sqlite3.connect(db_path)
        # except Exception as e:
        #     raise Exception(f"Can't connect to SQL: {e} in path {db_path}")
        conn.text_factory = str
        cursor = conn.cursor()
        table_meta['foreigns'], table_meta['foreign_key_columns'] = [], []
        for table in table_meta['table_names_original']:
            # Check table_info pragma
            cursor.execute("PRAGMA table_info({})".format(table.lower()))
            for res in cursor.fetchall():
                # Format: (cid, column_name, column_type, notnull, default_value, pk)
                _, column_name, column_type, _, _, pk = res
                key = f"{table.lower()}.{column_name.lower()}"
                assert key in table_meta['column_names_mapping'].keys()
                # Hard-code to remove ``(XX)`` in 'varchar' type
                normalized_type = re.sub(r'\(.*?\)', '', column_type)
                if normalized_type and normalized_type in COLUMN_TYPE_MAP.keys():
                    # Rewrite type by using database metadata as the first priority
                    table_meta['column_names_types_mapping'][key] = COLUMN_TYPE_MAP[normalized_type.lower()]
                if pk > 0 and column_name.lower() not in table_meta['primaries'][table.lower()]:
                    table_meta['primaries'][table.lower()].add(column_name.lower())
            # Check foreign_key_list pragma
            cursor.execute("PRAGMA foreign_key_list({})".format(table))
            for res in cursor.fetchall():
                # Format: (id, column_seq, foreign_table, local_column, foreign_column, on_update, on_delete, match)
                _, _, foreign_table, local_column, foreign_column, _, _, _ = res
                table_meta['foreign_key_columns'].append(f'{table.lower()}.{local_column.lower()}')
                table_meta['foreigns'].append(f'{table.lower()}-{foreign_table.lower()}')
                table_meta['foreign_column_pairs'][f'{table.lower()}.{local_column.lower()}'] = f'{foreign_table.lower()}.{foreign_column.lower()}'
                table_meta['foreign_column_pairs'][f'{foreign_table.lower()}.{foreign_column.lower()}'] = f'{table.lower()}.{local_column.lower()}'
    except:
        table_meta['foreigns'] = table_meta['foreigns_bak']
        table_meta['foreign_key_columns'] = table_meta['foreign_key_columns_bak']
        table_meta['foreign_column_pairs'] = table_meta['foreign_column_pairs_bak']

    return table_meta

def find_relationship_tables(schema):
    res = {}

    G = nx.DiGraph()
    for t in schema['table_names_original']:
        G.add_node(t.lower())
    
    for fk in schema['foreigns']:
        end, start = fk.split('-')
        G.add_edge(start, end)
    
    relation_nodes = [node for node in G.nodes if G.out_degree(node) == 0 and G.in_degree(node) == 2]
    for n in relation_nodes:
        res[n] = [nn for nn in G.predecessors(n)]

    return res

def is_consistent_columns(db_id, db_dir, g_table, col1, col2):
    values = read_table_column_values(db_id, db_dir, g_table, col1, col2)
    value_dict = defaultdict(set)
    for value in values:
        v1, v2 = value
        value_dict[v1].add(v2)
    
    if any(len(v) > 1 for v in value_dict.values()): return False

    return True

def is_same_type_columns(db_id, db_dir, col1, col2, schema):
    match = False
    values1 = read_table_column_values(db_id, db_dir, col1.split('.')[0].lower(), col1)
    values2 = read_table_column_values(db_id, db_dir, col2.split('.')[0].lower(), col2)
    if not values1 or not values2 or schema['column_names_types_mapping'][col1] == schema['column_names_types_mapping'][col2]:
        # any(v in values2 for v in values1): 
        match = True
    # Hard-code fix `number but text type` string-similar columns
    # elif any(t in col2.lower().split('.')[1].split('_') for t in col1.lower().split('.')[1].split('_')) or \
    #     any(t in col1.lower().split('.')[1].split('_') for t in col2.lower().split('.')[1].split('_')):
    else:
        values1 = [str(v[0]) for v in values1]
        values2 = [str(v[0]) for v in values2]
        if any(v in values2 for v in values1): 
            match = True

    return match


def is_consistent_schema_database(db_id: str, dataset_path: str, schema_path: str):
    # Get the content from schema file
    table_meta = defaultdict(dict)
    dbs_json_blob = json.load(open(schema_path, "r"))
    table_meta.update([db for db in dbs_json_blob if db_id == db['db_id']][0])
    for column, text in zip(table_meta['column_names_original'][1:], table_meta['column_names'][1:]):
        table_id, column_name = column
        _, column_text = text
        key = f"{table_meta['table_names_original'][table_id].lower()}.{column_name.lower()}"
        table_meta['column_names_mapping'][key] = column_text.lower()
    # Get the columns/tables from database file
    try:
        db_path = os.path.join(dataset_path, db_id, db_id + ".sqlite")
        conn = sqlite3.connect(db_path)
    except:
        return False
    conn.text_factory = str
    cursor = conn.cursor()
    for table in table_meta['table_names_original']:
        # Check table_info pragma
        cursor.execute("PRAGMA table_info({})".format(table.lower()))
        results = cursor.fetchall()
        # Table inconsistency
        if not results: return False
        db_table_columns = []
        for res in results:
            # Format: (cid, column_name, column_type, notnull, default_value, pk)
            _, column_name, _, _, _, _ = res
            key = f"{table.lower()}.{column_name.lower()}"
            db_table_columns.append(key)
        schema_table_columns = [k for k in table_meta['column_names_mapping'].keys() if f"{table.lower()}." in k]
        # Colum inconsistency
        if any(sc not in db_table_columns for sc in schema_table_columns):
            return False

    return True
