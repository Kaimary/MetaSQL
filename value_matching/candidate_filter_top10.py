import re
import click
import json
import copy
from tqdm import tqdm
from copy import deepcopy
from nltk import WordNetLemmatizer

from value_matching.spider_db_context import SpiderDBContext, is_number
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from spider_utils.utils import is_consistent_columns, is_same_type_columns, read_single_dataset_schema_from_database
from spider_utils.evaluation.evaluate1 import rebuild_sql, build_foreign_key_map_from_json, Evaluator
from utils.sql_utils import sql_nested_query_tmp_name_convert, sql_string_format

LEMMATIZER = WordNetLemmatizer()

def all_number(cur_value_set):
    flag = True
    for i in cur_value_set:
        if not is_number(i):
            flag = False
    return flag

def nl_reader(nl_file_path, reranker_input_file):
    nl_file = open(nl_file_path, 'r')
    data = json.load(nl_file)
    nl_list = []
    db_id_list = []
    with open(reranker_input_file, 'r') as f:
        json_obj = json.load(f)
        for ex in json_obj:
            index = ex['index']
            nl_list.append(data[index]['question'])
            db_id_list.append(data[index]['db_id'])
    return nl_list, db_id_list

def candidate_reader(candidates_file_path):
    candidate_list = []
    with open(candidates_file_path, 'r') as f_in:
        tmp = []
        for line in f_in.readlines():
            line = line.strip()
            if line == '':
                candidate_list.append(copy.deepcopy(tmp))
                tmp.clear()
            else:
                tmp.append(line)
        if len(tmp) != 0:
            candidate_list.append(copy.deepcopy(tmp))
    return candidate_list

def get_all_filter_column(sql_dict):
    """
    get all filter conditions from one sql dict
    @param sql_dict: sql rebuilt
    @return: filter columns set
    """
    # no a sql dict, no column found
    if not isinstance(sql_dict, dict):
        return {}
    filter_columns = list()
    # add every condition's col into set
    for condition in sql_dict['where']:
        # pass conjunction
        if condition == 'and' or condition == 'or':
            continue
        else:
            filter_columns.append(condition[2][1][1].strip('_'))  # add column
            filter_columns.extend(get_all_filter_column(condition[3]))  # recursively add column from nested
    # recursively handle IUE
    filter_columns.extend(get_all_filter_column(sql_dict['intersect']))
    filter_columns.extend(get_all_filter_column(sql_dict['union']))
    filter_columns.extend(get_all_filter_column(sql_dict['except']))
    return filter_columns

def heuristic_filter(candidates, db_id, schema, tables_file, db_dir, kmaps, pred, nl=None):
    if len(candidates) < 2: return candidates

    evaluator = Evaluator()
    top2_pos = 1
    while (top2_pos < len(candidates) and \
           evaluator.eval_exact_match(
                deepcopy(rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(candidates[0]), kmaps, tables_file, rebuild_col=False)), 
                deepcopy(rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(candidates[top2_pos]), kmaps, tables_file, rebuild_col=False))) == 1
            ): top2_pos += 1
    # If all are the same sqls, skip the heuristics here
    if top2_pos != len(candidates):
        # try:
        #     top1_sql  = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(candidates[0]), kmaps, tables_file, rebuild_col=False)
        #     top2_sql  = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(candidates[top2_pos]), kmaps, tables_file, rebuild_col=False)
        #     partial_scores = evaluator.eval_partial_match(deepcopy(top1_sql), deepcopy(top2_sql))
        #     match, exclude_select_partial_match = True, True
        #     exclue_partials = ['select', 'select(no AGG)']
        #     for key, score in partial_scores.items():
        #         if score['f1'] != 1:
        #             match = False
        #             if key not in exclue_partials:
        #                 exclude_select_partial_match = False
        #     # 1. Original prediction priority heuristic
        #     if candidates[top2_pos] == sql_string_format(pred) and evaluator.eval_hardness(top1_sql) != 'easy': # and exclude_select_partial_match:
        #         print("Original prediction priority heuristic")
        #         candidates[0], candidates[top2_pos] = candidates[top2_pos], candidates[0]
        #     # 2. COUNT-using-star heuristic
        #     elif exclude_select_partial_match and partial_scores['select']['label_total'] == 1 and partial_scores['select']['pred_total'] == 1:
        #         _, sels = top1_sql['select']
        #         agg_op1, col_unit1 = sels[0]
        #         _, sels2 = top2_sql['select']
        #         agg_op2, col_unit2 = sels2[0]
        #         if agg_op1 == agg_op2 and agg_op1 == 3 and col_unit1[1][1] != '__all__' and col_unit2[1][1] == '__all__':
        #             print("COUNT-using-star heuristic")
        #             candidates[0], candidates[top2_pos] = candidates[top2_pos], candidates[0]
        # except:
        #     pass
        # top1_sql  = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(candidates[0]), kmaps, tables_file, rebuild_col=False)
        pred_sql = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(pred), kmaps, tables_file, rebuild_col=False)
        if candidates[0] != sql_string_format(pred) and evaluator.eval_hardness(pred_sql) == 'extra':
            print("Original prediction priority heuristic (extra)")
            candidates[0] = pred
            
    # 1. Invalid-SQL heuristic
    # a) Duplicated projections
    # b) Group-projection consistence
    # c) predicate (subquery) column consistence 
    fix_pattern = r"([;:'\",<.>/?!@#$%^&*\(\)_+])"
    nl = re.sub(fix_pattern, r' \1 ', nl)
    nl = ' '.join(nl.split())
    cursor = -1 # Point to the last position that has already checked as `invalid`
    while (cursor + 1 < len(candidates)):
        invalid = False
        checking_item = candidates[cursor+1]
        try:
            sql  = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(checking_item), kmaps, tables_file, rebuild_col=False)
            #'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
            #'groupBy': [col_unit1, col_unit2, ...]
            _, sels = sql['select']
            s_cols = []
            for t in sels:
                agg_id, val_unit = t
                _, col_unit1, _ = val_unit
                _, col, _ = col_unit1
                s_col = f'{agg_id} {col}'
                if s_col in s_cols: invalid = True
                else: s_cols.append(s_col)
                # s_cols.append(s_col)

            if sql['groupBy']:
                _, g_col, _ = sql['groupBy'][0]
                g_col = g_col.strip('_')
                # If group-column is not the same as any select-columns
                # Check if group-column is `value-consistent` with any selelct-columns
                if all(g_col != sc.split()[1].strip('_') for sc in s_cols if sc.split()[0] == '0'):
                    mismatch_sels = [sc.split()[1].strip('_') for sc in s_cols if sc.split()[0] == '0']
                    match = False
                    for col2 in mismatch_sels:
                        g_table = g_col.split('.')[0]
                        col1 = g_col
                        if g_table != col2.split('.')[0]:
                            if g_col not in schema['foreign_column_pairs'].keys() or schema['foreign_column_pairs'][g_col].split('.')[0] != col2.split('.')[0]: continue
                            pair_column = schema['foreign_column_pairs'][g_col]
                            g_table = pair_column.split('.')[0]
                            col1 = pair_column
                        if is_consistent_columns(db_id, db_dir, g_table, col1, col2): match = True
                    if not match: invalid = True

            if sql['where']:
                for cond in sql['where'][::2]:
                    _, _, val_unit, val1, _ = cond
                    _, col_unit1, _ = val_unit
                    _, col1, _ = col_unit1
                    if isinstance(val1, dict):
                        _, sels = val1['select']
                        _, val_unit = sels[0]
                        _, col_unit1, _ = val_unit
                        _, col2, _ = col_unit1
                        if not is_same_type_columns(db_id, db_dir, col1.strip('_'), col2.strip('_'), schema): 
                            invalid = True

            # if sql['where']
            # if 'JOIN' in checking_item and \
            #     not is_valid_join_path(checking_item, db_id, tables_file):
            #     invalid = True
        except:
            invalid = True
        if not invalid: break
        cursor += 1

    if cursor > -1 and cursor + 1 < len(candidates):
        print("Invalid-SQL heuristic")
        candidates = candidates[cursor+1:] + candidates[:cursor+1]
        
    return candidates

def candidate_filter(candidates, db_id, db_context, dataset_path, kmaps):
    cur_column_list = []
    cur_value_set = set()

    entities = db_context.get_entities_from_question(db_context.string_column_mapping)
    cur_value_set.clear()
    cur_column_list.clear()

    for ent in entities:
        if is_number(ent[0].split(':')[-1]) or ent[0].split(':')[-1] == '' or ent[0].split(':')[-1] == 'd':
            continue
        cur_value_set.add(ent[0].split(':')[-1])
        col_tmp = set()
        for col in ent[1]:
            col_tmp.add(col.split(':')[-2] + '.' + col.split(':')[-1])
        cur_column_list.append(col_tmp)

    if cur_column_list:
        cur_candidates = []
        cur_del_num = 0
        del_index_list = []
        # kmaps = build_foreign_key_map_from_json(tables_file)

        for candidate in candidates:
            append_flag = True
            try:
                sql_dict = rebuild_sql(db_id, dataset_path, sql_nested_query_tmp_name_convert(candidate), kmaps)
                sql_filter_cols = get_all_filter_column(sql_dict)
            except:
                sql_filter_cols = []
            for col_set in cur_column_list:
                check_col_flag = False
                for col in col_set:
                    if col in sql_filter_cols:
                        check_col_flag = True
                        sql_filter_cols.remove(col)
                        break
                if not check_col_flag:
                    append_flag = False
                    break
            if append_flag:
                cur_candidates.append(candidate)
            else:
                del_index_list.append(candidates.index(candidate))
                cur_del_num += 1
        if cur_del_num == 5:
            cur_candidates = candidates
        return cur_candidates
    else:
        return candidates


@click.command()
@click.argument("nl_file_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("model_output_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("reranker_input_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("candidates_file_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("tables_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("database_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("pred_sql_path", type=click.Path(exists=False, dir_okay=False))
@click.argument("pred_sql_path_top_k", type=click.Path(exists=False, dir_okay=False))
def main(nl_file_path, model_output_file, reranker_input_file, candidates_file_path, tables_file, database_dir, pred_sql_path, pred_sql_path_top_k):
    lgesql = open(model_output_file, 'r')
    preds = [ex.strip() for ex in lgesql.readlines()]
    tokenizer = SpacyTokenizer()
    pred_sql_list = []
    nl_list, db_id_list = nl_reader(nl_file_path, reranker_input_file)
    candidate_list = candidate_reader(candidates_file_path)
    kmaps = build_foreign_key_map_from_json(tables_file)
    # filter loop, return pred sql list
    schema = read_single_dataset_schema_from_database(db_id_list[0], database_dir, tables_file)
    db_context = SpiderDBContext(
        db_id_list[0],
        nl_list[0],
        tokenizer,
        tables_file,
        database_dir
    )
    i = 0
    for nl, candidates, db_id, pred in tqdm(zip(nl_list, candidate_list, db_id_list, preds)):
        if db_context.db_id != db_id:
            schema = read_single_dataset_schema_from_database(db_id, database_dir, tables_file)
            db_context = SpiderDBContext(
                db_id,
                nl,
                tokenizer,
                tables_file,
                database_dir
            )
        db_context.change_utterance(nl)
        candidates = candidate_filter(candidates, db_id, db_context, database_dir, kmaps)
        candidates = heuristic_filter(candidates, db_id, schema, tables_file, database_dir, kmaps, pred, nl)
        i += 1
        pred_sql_list.append(candidates)

    # 3.pred sql writer
    with open(pred_sql_path, 'w') as f_out:
        for sqls in pred_sql_list:
            f_out.write(sqls[0] + '\n')
    with open(pred_sql_path_top_k, 'w') as f_out_1:
        for sqls in pred_sql_list:
            for sql in sqls:
                f_out_1.write(sql + '\n')
            f_out_1.write('\n')

if __name__ == '__main__':
    main()
