import os, json
from copy import deepcopy
from tqdm import tqdm

from evaluation import build_foreign_key_map_from_json
from utils.spider_utils.evaluation.evaluate import Evaluator, rebuild_sql
from utils.spider_utils.evaluation.process_sql import get_schema, get_schema_from_json, tokenize
from utils.spider_utils.utils import disambiguate_items2, read_single_dataset_schema
from utils.sql_utils import sql_nested_query_tmp_name_convert

def main(test_file, model_output_file, metaor_output_file, tables_file, db_dir):
    schema = {}
    table = {}
    table_dict = {}
    dataset = open(test_file, 'r')
    dataset_json = json.load(dataset)
   
    metaor = open(metaor_output_file)
    outputs = [line.strip() for line in metaor.readlines()]
    hit = 0
    pos = 0
    evaluator = Evaluator()
    kmaps = build_foreign_key_map_from_json(tables_file)
    model_output = open(model_output_file)
    lines = model_output.readlines()
    for idx, output in tqdm(enumerate(outputs)):
        output_lenth = len(output.split(', '))
        db_id = dataset_json[idx]['db_id']
        gold = dataset_json[idx]['query']
        if db_id not in schema:
            db_file = os.path.join(db_dir, db_id, db_id + ".sqlite")
            if not os.path.isfile(db_file): s = get_schema_from_json(db_id, tables_file)
            else: s = get_schema(db_file)
            _, t, td = read_single_dataset_schema(tables_file, db_id)   
            schema[db_id] = s
            table[db_id] = t
            table_dict[db_id] = td

        g_sql = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(gold), kmaps, tables_file)
        for sql in lines[pos: pos+output_lenth]:
            if sql.strip() == 'sql placeholder': 
                continue
            try:
                toks, _, _ = disambiguate_items2(tokenize(sql.strip()), schema[db_id], table[db_id], allow_aliases=False)
                sql = ' '.join(toks)
                sql = sql.replace('@', '.')
                # sql = sql_string_format(sql)
                # sql = fix_missing_join_condition(sql, db_id, tables_file)
                sql = sql_nested_query_tmp_name_convert(sql)
                p_sql = rebuild_sql(db_id, db_dir, sql, kmaps, tables_file)
                if evaluator.eval_exact_match(deepcopy(p_sql), deepcopy(g_sql)) == 1:
                    hit += 1
                    break
            except:
                pass
        pos += output_lenth

    print(f"{hit}/1034")

if __name__ == "__main__":
    test_file = "data/dev.json"
    model_output_file = "output/meta_preds.txt"
    metaor_output_file = "output/metadata.txt"
    tables_file = "data/tables.json"
    db_dir = "data/database"
    main(test_file, model_output_file, metaor_output_file, tables_file, db_dir)