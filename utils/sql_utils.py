import os
import json
from collections import defaultdict
from copy import deepcopy
from typing import DefaultDict

from utils.spider_utils.evaluation.evaluate import Evaluator
from utils.spider_utils.evaluation.evaluate import build_foreign_key_map_from_json
from utils.spider_db_context import is_number

SQL_KEYWORDS = ['SELECT', 'FROM', 'JOIN', 'ON', 'AS', 'WHERE', 'GROUP', 'BY', 'HAVING', \
    'ORDER', 'LIMIT', 'INTERSECT', 'UNION', 'EXCEPT', 'NOT', 'BETWEEN', 'IN', 'LIKE', 'IS', \
    'DISTINCT', 'DESC', 'ASC', 'AND', 'OR']
AGG_OPS = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']

# Compare the SQL structures and calculate a similarity score between the two.
def calculate_similarity_score(g_sql, p_sql):
    evaluator = Evaluator()
    total_score = 10.0
    # We first check the sources of the two sqls, we assume it dominants the similarity score.
    if len(g_sql['from']['table_units']) > 0:
        label_tables = sorted(g_sql['from']['table_units'])
        pred_tables = sorted(p_sql['from']['table_units'])
        if label_tables != pred_tables:
            total_score -= 1.0
        elif len(g_sql['from']['conds']) > 0:
            label_joins = sorted(g_sql['from']['conds'], key=lambda x: str(x))
            pred_joins = sorted(p_sql['from']['conds'], key=lambda x: str(x))
            if label_joins != pred_joins:
                total_score -= 0.5
    partial_scores = evaluator.eval_partial_match(deepcopy(p_sql), deepcopy(g_sql))
    # Next we use 7 of 10 categories from partial scores to do the comparison: 
    # 1)select 2)where 3)group 4)order 5)and/or 6)IUE 7)keywords
    for category, score in partial_scores.items():
        if score['f1'] != 1:
            if category == "keywords":
                total_score -= 5.0
            elif category == "select":
                total_score -= 1.0
            elif category == "where":
                total_score -= 1.0
            elif category == "group":
                total_score -= 1.0
            elif category == "order":
                total_score -= 1.0
            elif category == "and/or":
                total_score -= 0.5
            elif category == "IUEN":
                total_score -= 1.0

    return total_score

# Compare the SQL structures and calculate a similarity score between the two.
def calculate_similarity_score1(g_sql, p_sql):
    evaluator = Evaluator()
    total_score = 10.0
    if len(g_sql['from']['table_units']) > 0:
        label_tables = sorted(g_sql['from']['table_units'])
        pred_tables = sorted(p_sql['from']['table_units'])
        if label_tables != pred_tables:
            total_score -= 1.0

    partial_scores = evaluator.eval_partial_match(deepcopy(p_sql), deepcopy(g_sql))
    # Next we use 6 of 10 categories from partial scores to do the comparison: 
    # 1)select 2)where 3)group 4)order 5)IUE 6)keywords
    for category, score in partial_scores.items():
        if score['f1'] != 1:
            if category == "keywords":
                total_score -= 3.0
            elif category == "select":
                total_score -= 3.0
            elif category == "where":
                total_score -= 2.0
            elif category == "group":
                total_score -= 2.0
            elif category == "order":
                total_score -= 2.0
            elif category == "IUEN":
                total_score -= 2.0

    return total_score


def get_low_confidence_generalized_data(
    serialization_file, tables_file,
    trial=100, rewrite=False, overwrite=False, mode='train'
):
    """
    Get the generalized sql-dialects from a low-confidence query
    The generalization process will serialize the generalized data into local files.
        1. If the query contains low-confidence marks, execute the generation and serialize the data into file and output
        2. If the query has no low-confidence mark or fails to generate, serialize into an empty file and skip


    :param dataset_name: the name of NLIDB benchmark
    :param dataset_file: the train/dev/test file
    :param model_output_file: the corresponding inferred results of SODA seq2seq model of the datasest file
    :param tables_file: database schema file
    :param db_dir: the diretory of databases
    :param overwrite: if overrite existing serialization files
    :param mode: train/dev/test mode
    
    :return: serialize the data into local files
    """
    global kmaps
    kmaps = build_foreign_key_map_from_json(tables_file)

    sqls = []
    dialects = []
    if not os.path.exists(serialization_file) or overwrite:
        # Create an empty serialization file first
        datafile = open(serialization_file, 'w')
        # gen2 = GeneratorV2(dataset_file, tables_file, db_dir, trial=trial)
        # gen2.load_database(db_id)
        # if not inferred_query_with_marks: return sqls, dialects
        # # For training purpose, we randomize to add low-confidence marks to augment the training size
        # if mode =="train" and '@' not in inferred_query_with_marks:
        #     tokens = inferred_query_with_marks.split()
        #     tokens1 = [1 for t in tokens if t not in ["DESC", "ASC", "ON", "JOIN", "HAVING", "'terminal'", "BY", "DISTINCT"]]
        #     max = 6 if len(tokens1) > 6 else len(tokens1)
        #     num = random.randint(2, max)
        #     while num:
        #         i = random.randint(0, len(tokens) - 1)
        #         if '@' in tokens[i] or tokens[i] in ["DESC", "ASC", "ON", "JOIN", "HAVING", "'terminal'", "BY", "DISTINCT"]: continue
        #         tokens[i] = f'@{tokens[i]}'
        #         num -= 1
        #     inferred_query_with_marks = ' '.join(tokens)
        # try: sqls_ = gen2.generate(inferred_query_with_marks, inferred_query)
        # except:
        #     print(f"ERR in SQLGenV2 - {db_id}: {inferred_query_with_marks}")
        #     os.remove(serialization_file)
        #     return sqls, dialects
        # for sql in sqls_:
        #     try:
        #         sql = sql_nested_query_tmp_name_convert(sql)
        #         sql = use_alias(sql)
        #         _, sql_dict, schema_ = disambiguate_items2(tokenize(sql), schema, table, allow_aliases=False)
        #         # dialect = convert_sql_to_dialect(sql_dict, table_dict, schema_)
        #         dialect = sql
        #         sqls.append(sql)
        #         dialects.append(dialect)
        #     except: pass
        # # Invalid sql
        # if not sqls: return sqls, dialects
        # # If only one sql left, check if it is the same with original one, 
        # # since the generation may revise the syntax/semantics error in the orignal sql
        # if len(sqls) == 1:
        #     p_sql  = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(inferred_query), kmaps, tables_file)
        #     p_sql1 = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(sqls[0]), kmaps, tables_file)
        #     if evaluator.eval_exact_match(deepcopy(p_sql), deepcopy(p_sql1)) == 1: return [], []
        # # Serialize the genration results
        # # Each line includes a sql and the corresponding dialect
        # for sql, dialect in zip(sqls, dialects):
        #     # write line to output file
        #     line = f'{sql}\t{dialect}\n'
        #     datafile.write(line)
        # datafile.close()
    # Read from the existing serialization file if exists
    else:
        # Skip empty serizalization 
        if os.stat(serialization_file).st_size == 0: return sqls, dialects
        all_lines = []
        datafile = open(serialization_file, 'r')
        for line in datafile.readlines():
            sql, dialect = line.split('\t')
            sql = ' '.join(sql.strip().split())
            if rewrite:
                sql = sql_nested_query_tmp_name_convert(sql)
                # dialect = convert_sql_to_dialect(sql_dict, table_dict, schema_)
                dialect = sql
                line = sql + '\t' + dialect + '\n'
                all_lines.append(line)
            if sql not in sqls: 
                sqls.append(sql)
                dialects.append(dialect.strip())
        if rewrite:
            datafile = open(serialization_file, 'w')
            datafile.writelines(all_lines)
            datafile.close()
    
    return sqls, dialects

import re
def sql_dialect_format(sql, cols_dict):
    """
    :param sql: sql string
    :param cols_dict: orginal column name-annotation map
    :return: formatted sql string
    
    Format sql string by removing useless parts for semantics capture.
    1. Remove table reference from columns (e.g., `table.column`=>`column`)
    2. Replace column name with its annotation
    3. Remove alias
    """

    sql = re.sub(' AS T\d', '', sql)
    toks = sql.split()
    for idx, t in enumerate(toks):
        if is_number(t) or '.' not in t: continue
        
        if t[-1] == ',': t = t[: -1]
        while t[-1] == ')': t = t[: -1]
        full_column_name = t[t.find("(")+1:]
        if not full_column_name.split('.')[1]: continue
        column_txt = cols_dict[full_column_name.split('.')[1].lower()]
        toks[idx] = toks[idx].replace(full_column_name, column_txt)
    
    return ' '.join(toks)


def sql_string_format(sql):
    """
    Format SQL string 
    All sql keywords are upper cased and lower-case other tokens

    :param sql: the inferred sql string
    :return the sql string with all sql keywords uppercased
    """
    sql = re.sub("'terminal'", "\"value\"", sql)
    sql = sql.lower()

    pattern = re.compile(r'([\(]*)([\s]*)select ')
    sql = re.sub(pattern, r'\1\2SELECT ', sql)
    pattern = re.compile(r'([\(]*)([\s]*)intersect ')
    sql = re.sub(pattern, r'\1\2INTERSECT ', sql)
    pattern = re.compile(r'([\(]*)([\s]*)union ')
    sql = re.sub(pattern, r'\1\2UNION ', sql)
    pattern = re.compile(r'([\(]*)([\s]*)except ')
    sql = re.sub(pattern, r'\1\2EXCEPT ', sql)

    pattern = re.compile(r' count([\s]*)(\()')
    sql = re.sub(pattern, r' COUNT\1\2', sql)
    pattern = re.compile(r' sum([\s]*)(\()')
    sql = re.sub(pattern, r' SUM\1\2', sql)
    pattern = re.compile(r' avg([\s]*)(\()')
    sql = re.sub(pattern, r' AVG\1\2', sql)
    pattern = re.compile(r' max([\s]*)(\()')
    sql = re.sub(pattern, r' MAX\1\2', sql)
    pattern = re.compile(r' min([\s]*)(\()')
    sql = re.sub(pattern, r' MIN\1\2', sql)

    sql = re.sub(' from ', ' FROM ', sql)
    sql = re.sub(' where ', ' WHERE ', sql)
    sql = re.sub(' join ', ' JOIN ', sql)
    sql = re.sub(' order by ', ' ORDER BY ', sql)
    sql = re.sub(' group by ', ' GROUP BY ', sql)
    sql = re.sub(' having ', ' HAVING ', sql)
    sql = re.sub(' limit ', ' LIMIT ', sql)
    sql = re.sub(' between ', ' BETWEEN ', sql)
    sql = re.sub(' like ', ' LIKE ', sql)
    sql = re.sub(' not ', ' NOT ', sql)
    sql = re.sub(' in ', ' IN ', sql)
    sql = re.sub(' or ', ' OR ', sql)
    sql = re.sub(' and ', ' AND ', sql)
    sql = re.sub(' desc', ' DESC', sql)
    sql = re.sub(' asc', ' ASC', sql)
    # agg_op = ""
    # for idx, tok in enumerate(toks):
    #     if tok.replace(')', '').upper() in SQL_KEYWORDS and not agg_op: toks[idx] = tok.upper()
    #     elif tok.upper() in AGG_OPS: 
    #         agg_op = tok.upper()
    #         toks[idx] = ""
    #     elif agg_op and tok == ')':
    #         agg_op += ')'
    #         toks[idx] = agg_op
    #         agg_op = ""
    #     elif agg_op: 
    #         if tok.upper() in SQL_KEYWORDS: tok = tok.upper() + ' '
    #         agg_op += tok
    #         toks[idx] = ""
    #     elif '(' in tok:
    #         parts = tok.split('(')
    #         # (SELECT
    #         if not parts[0]: toks[idx] = f'({parts[1].upper()}'
    #         # AGG(col)
    #         else:
    #             agg = parts[0]
    #             assert agg.upper() in AGG_OPS
    #             other = parts[1]
    #             # AGG(DISTINCT
    #             if other.upper() in SQL_KEYWORDS: toks[idx] = f'{agg.upper()}({other.upper()}'
    #             else: toks[idx] = f'{agg.upper()}({other}'
    
    # return ' '.join([t for t in toks if t])
    return sql

def sql_nested_query_tmp_name_convert(sql: str, nested_level=0, sub_query_token='S') -> str:
    sql = sql.replace('(', ' ( ')
    sql = sql.replace(')', ' ) ')
    tokens = sql.split()
    select_count = sql.lower().split().count('select')
    level_flag = sub_query_token * nested_level

    # recursive exit
    if select_count == 1:
        # need to fix the last level's tmp name
        res = sql
        if nested_level:
            # log all tmp name
            tmp_name_list = set()
            for i in range(len(tokens)):
                # find tmp name
                if tokens[i].lower() == 'as':
                    tmp_name_list.add(tokens[i + 1])
                # convert every tmp name
            for tmp_name in tmp_name_list:
                res = res.replace(f' {tmp_name}', f' {level_flag}{tmp_name}')
        return res

    # for new sql's token
    new_tokens = list()
    bracket_num = 0
    i = 0
    # iter every token in tokens
    while i < len(tokens):
        # append ordinary token
        new_tokens.append(tokens[i])
        # find a nested query
        if tokens[i] == '(' and tokens[i + 1].lower() == 'select':
            nested_query = ''
            bracket_num += 1
            left_bracket_position = i + 1
            # in one nested query
            while bracket_num:
                i += 1
                if tokens[i] == '(':
                    bracket_num += 1
                elif tokens[i] == ')':
                    bracket_num -= 1
                # to the end of the query
                if bracket_num == 0:
                    # format new nested query and get the tokens
                    nested_query = ' '.join(tokens[left_bracket_position: i])
                    nested_query = sql_nested_query_tmp_name_convert(nested_query, nested_level + 1)
            # new sql's token log
            new_tokens.append(nested_query)
            # append the right bracket
            new_tokens.append(tokens[i])
        # IUE handle
        elif tokens[i].lower() in {'intersect', 'union', 'except'}:
            nested_query = ' '.join(tokens[i + 1:])
            nested_query = sql_nested_query_tmp_name_convert(nested_query, nested_level + 10)
            new_tokens.append(nested_query)
            i += 9999
        i += 1
    # format the new query
    res = ' '.join(new_tokens)
    if nested_level:
        # log all tmp name
        tmp_name_list = set()
        for i in range(len(new_tokens)):
            # find tmp name
            if new_tokens[i].lower() == 'as':
                tmp_name_list.add(new_tokens[i + 1])
            # convert every tmp name
        for tmp_name in tmp_name_list:
            res = res.replace(f' {tmp_name}', f' {level_flag}{tmp_name}')

    return res

IUE = ['', ' INTERSECT ', ' EXCEPT ', ' UNION ']
IUE_TOKENS = ['INTERSECT', 'EXCEPT', 'UNION']
CLS_TOKENS = ['SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER']

def split_into_simple_sqls(sql):
    """
    Split SQL with Set Operators(intersect/except/union) into simple SQLs.

    :param sql: complex sql string
    :return: main sql and iue-type sqls if exists

    #TODO support multiple intersects/excepts/unions
    """

    ssql = ""
    intersect_ = ""
    except_ = ""
    union_ = ""
    
    split_indice = []
    tokens = sql.split()
    left_brackets = 0
    for idx, t in enumerate(tokens):
        if '(' in t: left_brackets += 1
        if ')' in t: left_brackets -= 1
        if any(tt in t for tt in IUE_TOKENS) and left_brackets == 0: split_indice.append(idx)
    split_indice.append(len(tokens))

    start = 0
    for i in split_indice:
        if start == 0: ssql = ' '.join(tokens[start: i])
        elif IUE_TOKENS[0] in tokens[start]: intersect_ = ' '.join(tokens[start: i]) 
        elif IUE_TOKENS[1] in tokens[start]: except_ = ' '.join(tokens[start: i]) 
        elif IUE_TOKENS[2] in tokens[start]: union_ = ' '.join(tokens[start: i]) 
        start = i

    return ssql, intersect_, except_, union_

def split_into_clauses(sql):
    """
    Split SQL into clauses(select/from/where/groupby/orderby).

    :param sql: sql string
    :return: sql clause strings
    """

    select_ = ""
    from_   = ""
    where_ = ""
    group_ = ""
    order_ = ""

    split_indice = []
    tokens = sql.split()
    left_brackets = 0
    for idx, t in enumerate(tokens):
        if '(' in t: left_brackets += 1
        if ')' in t: left_brackets -= 1
        if any(tt in t for tt in CLS_TOKENS) and left_brackets == 0:
            split_indice.append(idx)
    split_indice.append(len(tokens))

    assert split_indice[0] == 0
    start = 0
    for i in split_indice[1:]:
        if CLS_TOKENS[0] in tokens[start]: select_ = ' '.join(tokens[start: i])
        elif CLS_TOKENS[1] in tokens[start]: from_ = ' '.join(tokens[start: i]) 
        elif CLS_TOKENS[2] in tokens[start]: where_ = ' '.join(tokens[start: i]) 
        elif CLS_TOKENS[3] in tokens[start]: group_ = ' '.join(tokens[start: i]) 
        elif CLS_TOKENS[4] in tokens[start]: order_ = ' '.join(tokens[start: i]) 
        start = i
    
    return select_, from_, where_,  group_, order_


def add_join_conditions(from_, tables_file, db_id):
    def _find_shortest_path(start, end, graph):
        stack = [[start, []]]
        visited = set()
        while len(stack) > 0:
            ele, history = stack.pop()
            if ele == end:
                return history
            for node in graph[ele]:
                if node[0] not in visited:
                    stack.append((node[0], history + [(node[0], node[1])]))
                    visited.add(node[0])

    dbs_json_blob = json.load(open(tables_file, "r"))
    graph = defaultdict(list)
    table_list = []
    dbtable = {}
    for table in dbs_json_blob:
        if db_id == table['db_id']:
            dbtable = table
            for acol, bcol in table["foreign_keys"]:
                t1 = table["column_names"][acol][0]
                t2 = table["column_names"][bcol][0]
                graph[t1].append((t2, (acol, bcol)))
                graph[t2].append((t1, (bcol, acol)))
            table_list = [table for table in table["table_names_original"]]

    table_alias_dict = {}
    idx = 1

    tables = [t.lower() for t in from_.split() if t not in ['JOIN', 'FROM']]
    prev_table_count = len(tables)
    candidate_tables = []
    for table in tables:
        for i, table1 in enumerate(table_list):
            if table1.lower() == table:
                candidate_tables.append(i)
                break

    ret = ""
    after_table_count = 0
    if len(candidate_tables) > 1:
        start = candidate_tables[0]
        table_alias_dict[start] = idx
        idx += 1
        ret = "FROM {}".format(dbtable["table_names_original"][start].lower())
        after_table_count += 1
        try:
            for end in candidate_tables[1:]:
                if end in table_alias_dict:
                    continue
                path = _find_shortest_path(start, end, graph)
                # print("got path = {}".format(path))
                prev_table = start
                if not path:
                    table_alias_dict[end] = idx
                    idx += 1
                    ret = ""
                    continue
                for node, (acol, bcol) in path:
                    if node in table_alias_dict:
                        prev_table = node
                        continue
                    table_alias_dict[node] = idx
                    idx += 1
                    # print("test every slot:")
                    # print("table:{}, dbtable:{}".format(table, dbtable))
                    # print(dbtable["table_names_original"][node])
                    # print(dbtable["table_names_original"][prev_table])
                    # print(dbtable["column_names_original"][acol][1])
                    # print(dbtable["table_names_original"][node])
                    # print(dbtable["column_names_original"][bcol][1])
                    ret = "{} JOIN {} ON {}.{} = {}.{}".format(ret, dbtable["table_names_original"][node].lower(),
                                                                dbtable["table_names_original"][prev_table].lower(),
                                                                dbtable["column_names_original"][acol][1].lower(),
                                                                dbtable["table_names_original"][node].lower(),
                                                                dbtable["column_names_original"][bcol][1].lower())
                    after_table_count += 1
                    prev_table = node

        except:
            print("\n!!Exception in adding join conditions!!")

        return ret, prev_table_count == after_table_count
    else: return from_, True


def fix_missing_join_condition(sql, db_id, tables_file):
    if 'JOIN' not in sql or ' ON ' in sql: return sql

    new_sql = ""
    simple_sql_, intersect_, except_, union_ = split_into_simple_sqls(sql)

    for idx, s in enumerate([simple_sql_, intersect_, except_, union_]):
        if s:
            if idx > 0: s = ' '.join(s.split()[1:])
            if not s: new_sql += f'@@{IUE[idx].strip()} '
            else:
                scls, fcls, wcls,  gcls, ocls = split_into_clauses(s)
                fcls1, _ = add_join_conditions(fcls.replace('@', ''), tables_file, db_id)
                if not fcls1: return ""
                # retain low-confidence tags
                tokens = fcls.split()
                tokens1 = fcls1.split()
                for i, t in enumerate(tokens1):
                    for tt in tokens:
                        if '@' in tt and tt[1:] == t:
                            tokens1[i] = f'@{t}'
                fcls1 = ' '.join(tokens1)

                if ' JOIN ' in wcls: 
                    f_sp = wcls.index('FROM')
                    wcls_1 = wcls[:f_sp]
                    f_ep = len(wcls) - 1
                    if 'WHERE' in wcls[f_sp:]:
                        f_ep = wcls.rindex('WHERE')
                    elif 'GROUP' in wcls[f_sp:]:
                        f_ep = wcls.rindex('GROUP')
                    elif 'ORDER' in wcls[f_sp:]:
                        f_ep = wcls.rindex('ORDER')
                    wcls_2, _ = add_join_conditions(wcls[wcls.index('FROM'):f_ep].replace('@', ''), tables_file, db_id)
                    wcls_3 = wcls[f_ep:]
                    wcls = ' '.join([wcls_1, wcls_2, wcls_3])
                ss = ' '.join([scls, fcls1, wcls,  gcls, ocls])
                if idx > 0: new_sql += IUE[idx]
                new_sql += ss

    return new_sql

def add_values(candidates, db_context, schema):
    n_values = []
    s_values = []
    s_referred_columns = []

    # Find all detected entities
    entities = db_context.get_entities_from_question(db_context.string_column_mapping)
    for ent in entities:
        value = ent[0].split(':')[-1]
        if value in ['', 'd']: continue
        if is_number(value): 
            value1 = int(value) if '.' not in value else float(value)
            n_values.append(value1)
            continue # For numerical values, no need to record the referred columns
        else: s_values.append(value)
        cols = set()
        for col in ent[1]:
            cols.add(col.split(':')[-2] + '.' + col.split(':')[-1])
        s_referred_columns.append(cols)
    # Replace values with the corresponding recognized entities for each sql
    # 1) The replacement skips if inconsistent with the number of recognized entities and the number of `value` placeholders;
    # 2) First, replace all 1-1 matching placeholders with the corresponding entities
    # 3) Second, replace the rest of placeholders in order
    new_candidates = []
    for sql in candidates:
        tokens = sql.split()
        n_value_indices = []
        s_value_indices = []
        s_referred_columns1 = []
        for idx, token in enumerate(tokens):
            if token == '"value"':
                referred_col = tokens[idx - 2] if tokens[idx - 2] != '"value"' else tokens[idx - 4]
                if any(agg in referred_col for agg in ['*', 'AVG', 'MIN', 'MAX', 'COUNT', 'SUM']) or \
                    (referred_col.lower() in schema['column_names_types_mapping'].keys() and schema['column_names_types_mapping'][referred_col.lower()]) == 'number':
                        n_value_indices.append(idx)
                else:
                    s_value_indices.append(idx)
                    s_referred_columns1.append(referred_col)
                    
        if len(n_values) != len(n_value_indices) or len(s_values) != len(s_value_indices): 
            new_candidates.append(sql)
            continue
        
        # Find the 1-1-matching placeholders
        n_matched_indices = {}
        s_matched_indices = DefaultDict(list)
        if len(n_values) == 1: n_matched_indices[n_value_indices[0]] = n_values[0]
        for s_index, col in zip(s_value_indices, s_referred_columns1):
            for v, ref_cols in zip(s_values, s_referred_columns):
                if col in ref_cols: s_matched_indices[s_index].append(v)
        # Replace placeholders with values and remove the indices
        for k, v in n_matched_indices.items():
            tokens[k] = str(v)
            n_value_indices.remove(k)
        if len(s_matched_indices.keys()) < len(s_value_indices): # Skip the replacement of current sql
            new_candidates.append(sql)
            continue
        for k, v in s_matched_indices.items():
            if len(v) == 1:
                tokens[k] = f'"{v[0]}"'
                s_value_indices.remove(k)

        for n_index, v in zip(n_value_indices, n_values):
            tokens[n_index] = str(v)
        for s_index, v in zip(s_value_indices, s_values):
            tokens[s_index] = f'"{v}"'

        new_sql = ' '.join(tokens)
        new_candidates.append(new_sql)

    return new_candidates

def is_valid_join_path(sql, db_id, tables_file):
    if 'JOIN' not in sql: return True

    simple_sql_, intersect_, except_, union_ = split_into_simple_sqls(sql)
    for idx, s in enumerate([simple_sql_, intersect_, except_, union_]):
        if s:
            if idx > 0: s = ' '.join(s.split()[1:])
            else:
                _, fcls, wcls, _, _ = split_into_clauses(s)
                fcls1, valid = add_join_conditions(fcls, tables_file, db_id)
                if not fcls1 or not valid: return False

                if ' JOIN ' in wcls: 
                    f_sp = wcls.index('FROM')
                    f_ep = len(wcls) - 1
                    if 'WHERE' in wcls[f_sp:]:
                        f_ep = wcls.rindex('WHERE')
                    elif 'GROUP' in wcls[f_sp:]:
                        f_ep = wcls.rindex('GROUP')
                    elif 'ORDER' in wcls[f_sp:]:
                        f_ep = wcls.rindex('ORDER')
                    wcls_2, valid = add_join_conditions(wcls[wcls.index('FROM'):f_ep], tables_file, db_id)
                    if not wcls_2 or not valid: return False

    return True


def check_sql_fk_cols(sql, schema):
    exist = False
    _, cols = sql['select']

    for col in cols:
        agg_id, val_unit = col
        _, col_unit, _ = val_unit
        _, col_name, _ = col_unit
        col_name = col_name.strip('_')
        if col_name in schema['foreign_key_columns']: 
            exist = True

    return exist