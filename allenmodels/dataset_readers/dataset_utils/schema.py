# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 10:01
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : schema.py
# @Software: PyCharm
import json
from copy import deepcopy

DEFAULT_NAME = {
    'eq': 'equal',
    'like': 'like',
    'nlike': 'not like',
    'add': 'add',
    'sub': 'subtract',
    'nin': 'not in',
    'lte': 'less than or equal to',
    'lt': 'less than',
    'neq': 'not equal',
    'in': 'in',
    'gte': 'greater than or equal to',
    'gt': 'greater than',
    'And': 'and',
    'Or': 'or',
    'except': 'except',
    'union': 'union',
    'intersect': 'intersect',
    'Product': 'join',
    'Val_list': ',',
    'Orderby_desc': 'order in descending by',
    'Orderby_asc': 'order in ascending by',
    'Project': 'find',
    'Selection': 'which',
    'Limit': 'top',
    'Groupby': 'group by',
    'keep': 'keep',
    'min': 'minimum',
    'count': 'count',
    'max': 'maximum',
    'avg': 'average',
    'sum': 'sum',
    'Subquery': 'subquery',
    'distinct': 'distinct',
    'literal': '',
    'nan': 'not a number',
    'Table': 'table',
    'Value': 'value'
}

RESERVED_WORD = {
    'descending', 'intersect', 'sum', 'a', 'to', ',', 'join', 'and',
    'union', 'order', 'by', 'or', 'top', 'count', 'keep', 'group',
    'average', 'except', 'like', 'in', 'minimum', 'value', 'greater',
    'add', 'find', 'equal', 'subtract', 'number', 'not', 'than', 'less',
    'subquery', 'which', 'distinct', 'maximum', 'table', 'ascending'
}


def get_schema_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schema = {}
    for entry in data:
        db_mapping = {}
        # for t, t_o in zip(entry['table_names'], entry['table_names_original']):
        #     db_mapping[t_o.lower()] = t.lower()
        # for c, c_o in zip(entry['column_names'], entry['column_names_original']):
        #     if c_o[0] < 0:
        #         continue
        #     table_name = entry['table_names_original'][c_o[0]].lower()
        #     col_name = c[1].lower().replace(table_name.lower(), '').strip()
        #     db_mapping[f"{table_name}"] += f"|{col_name}"

        for t_o in entry['table_names_original']:
            db_mapping[t_o.lower()] = t_o.lower().replace(' ', '_')
        for c in entry['column_names']:
            if c[0] < 0:
                continue
            table_name = entry['table_names_original'][c[0]].lower()
            # col_name = c[1].lower().replace(table_name.lower(), '').strip()
            col_name = c[1].lower().strip()
            db_mapping[f"{table_name}"] += f"|{col_name}".replace(' ', '_')
        db_mapping.update(DEFAULT_NAME)
        schema[entry['db_id']] = deepcopy(db_mapping)

    return schema


def test_schema():
    print("\n\n[OUT] for test_schema:")
    schema = get_schema_from_json('datasets/spider/tables.json')
    print(schema['car_1'])
