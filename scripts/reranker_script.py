import random
import numpy as np
import json
import os
import click
import faiss
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

from sentence_transformers import SentenceTransformer
from utils.spider_db_context import SpiderDBContext
from utils.recall_checker_utils import RecallChecker
from utils.sql_utils import add_values, sql_nested_query_tmp_name_convert, sql_string_format
from utils.spider_utils.utils import read_single_dataset_schema_from_database, remove_conds
from utils.spider_utils.evaluation.process_sql import get_schema, get_schema_from_json
from utils.spider_utils.evaluation.evaluate import Evaluator, build_foreign_key_map_from_json, rebuild_sql
from configs.config import DIR_PATH, SERIALIZE_DATA_DIR, RETRIEVAL_MODEL_DIR, \
    RETRIEVAL_MODEL_EMBEDDING_DIMENSION

@click.command()
@click.argument("dataset_name", default="spider")
@click.argument("model_name", default="gap")
@click.argument("retrieval_model_name", default="nli-distilroberta-base-v2")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("model_output_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("tables_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("db_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("candidate_num", default=300)
@click.argument("mode", default="train")
@click.argument("debug", default=False)
@click.argument("output_file", type=click.Path(exists=False, dir_okay=False))
def main(
    dataset_name, model_name, retrieval_model_name, 
    dataset_file, model_output_file, tables_file, db_dir, 
    candidate_num, mode, debug, output_file
    ):
    """
    Generalize inferred queries with low-confidence marks and the corresponding dialects
    as the input for re-ranking model for further training/testing

    :param dataset_name: the name of NLIDB benchmark
    :param model_name: the seq2seq model name
    :param retrieval_model_name: the name of the trained retrieval model
    :param dataset_file: the train/dev/test file
    :param model_output_file: the corresponding inferred results of SODA seq2seq model of the datasest file
    :param tables_file: database schema file
    :param db_dir: the diretory of databases
    :param candidate_num: the filtered candiate number of the retrieval model
    :param mode: train/dev/test mode
    :param debug: debug mode
    :param output_file: the output file

    :return: a list of data as the input for reranking model
    """
    # Initialization
    if model_name == "gap":
        gap = open('bart_run_1_true_1-step41000-lc4.eval', 'r')
        data = json.load(gap)
        preds = [ex['predicted'] for ex in data['per_item']]
    elif model_name == "lgesql":
        lgesql = open(model_output_file, 'r')
        preds = [line.strip() for line in lgesql.readlines()]

    t_cols_dict = defaultdict(dict)
    dbs_json_blob = json.load(open(tables_file, "r"))
    for db in dbs_json_blob:
        db_id = db['db_id']
        for column_orig, column_txt in zip(db['column_names_original'][1:], db['column_names'][1:]):
            _, column_name = column_orig
            _, column_text = column_txt
            t_cols_dict[db_id][column_name.lower()] = column_text.lower()

    schema = {}
    table_dict = {}
    serialization_dir = f'{DIR_PATH}{SERIALIZE_DATA_DIR}/{model_name}/{mode}'
    if not os.path.exists(serialization_dir): os.makedirs(serialization_dir)
    kmaps = build_foreign_key_map_from_json(tables_file)
    evaluator = Evaluator()
    # Load the trained retrieval model
    embedder = SentenceTransformer(
        DIR_PATH + RETRIEVAL_MODEL_DIR.format(dataset_name) + '/' + retrieval_model_name)
    tokenizer = SpacyTokenizer()
    if debug: 
        # Statistics (debug purpose)
        # For correct inferred queries, the generation always hits as the original inferred query keeps. 
        # Therefore, the following two counts are only for incorrect inferred queries.
        checker = RecallChecker(dataset_file, tables_file, db_dir)
        hit_gen_num = 0 
        miss_gen_num = 0

    output = []
    with open(dataset_file, "r") as data_file:
        data = json.load(data_file)
        total_count = 0
        for index, (ex, pred) in tqdm(enumerate(zip(data, preds))):
            total_count += 1
            db_id = ex['db_id']
            if db_id not in schema:
                db_file = os.path.join(db_dir, db_id, db_id + ".sqlite")
                s = get_schema_from_json(db_id, tables_file) if not os.path.isfile(db_file) else get_schema(db_file)
                # _, t, td = read_single_dataset_schema(tables_file, db_id)   
                td = read_single_dataset_schema_from_database(db_id, db_dir, tables_file)
                schema[db_id] = s
                # table[db_id] = t
                table_dict[db_id] = td

                db_context = SpiderDBContext(
                    db_id,
                    ex['question'],
                    tokenizer,
                    tables_file,
                    db_dir
                )

            question = ex['question']
            serialization_file = f'{serialization_dir}/{index}.txt'
            sqls = []
            if os.stat(serialization_file).st_size != 0:
                for line in open(serialization_file, 'r').readlines():
                    sql, _ = line.split('\t')
                    sql = ' '.join(sql.strip().split())
                    if sql not in sqls: 
                        sqls.append(sql)

            if pred:
                pred = sql_string_format(pred)
                sqls.append(pred)

            sqls = [sql_string_format(remove_conds(sql)) for sql in sqls]
            hit = False
            # Check if the generation hits for incorrect inferred results
            if debug:
                try:
                    g_sql = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(ex['query']), kmaps, tables_file)
                    for sql in sqls:
                        p_sql = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(sql), kmaps, tables_file)
                        if evaluator.eval_exact_match(deepcopy(p_sql), deepcopy(g_sql)) == 1:
                            hit_gen_num += 1
                            hit = True
                            break
                except: pass
                if not hit:
                    miss_gen_num += 1
                    # print("<Generate but Miss>")
                    # print(f"{index} gold sql: {ex['query']}")
                    # print("===============================================================================================================================")

            # Make sure the generated number fixes to 100
            # while len(sqls) < candidate_num:
            #     # add the first sql repeately
            #     sqls.append(sqls[0])
            #     dialects.append(dialects[0])

            num = len(sqls) if len(sqls) < candidate_num else candidate_num
            # Get the top-k sql-dialect pairs
            question_embedding = embedder.encode(question)
            dialect_embeddings = embedder.encode(sqls)
            fidx = faiss.IndexFlatL2(int(RETRIEVAL_MODEL_EMBEDDING_DIMENSION))
            fidx.add(np.stack(dialect_embeddings, axis=0))
            _, indices = fidx.search(np.asarray(question_embedding).reshape(1, int(RETRIEVAL_MODEL_EMBEDDING_DIMENSION)), num)
            # candidate_dialects = [dialects[indices[0, idx]] for idx in range(0, num)]
            candidate_sqls = [sqls[indices[0, idx]] for idx in range(0, num)]
            # If the geneartion fits for incorrect inferred results or those correct inferred results
            # Check the precision for the retrieval model
            gold_sql_indices = []
            if mode == 'train' or mode == 'dev' or (mode == 'test' and debug and hit): 
                try:
                    gold_sql_indices = \
                        checker.check_add_candidategen_miss_sql(db_id, candidate_sqls, ex['query'], "True")
                except: continue

            # For training/validation purpose, add gold sql back if not exists in the candiates
            if mode == "train" or mode == "dev":
                if not gold_sql_indices:
                    try:
                        candidate_sqls.pop()
                        candidate_sqls.append(ex['query'])
                        gold_sql_indices.append(num-1)
                    except: continue
            
            while num < candidate_num:
                candidate_sqls.append(candidate_sqls[0])
                # candidate_dialects.append(dialects[0])
                if 0 in gold_sql_indices: gold_sql_indices.append(num)
                num += 1
            

            # Add values into sqls for ranking
            # kmaps = build_foreign_key_map_from_json(tables_file)
            # filter loop, return pred sql list
            try:
                db_context.change_utterance(question)
                candidates_with_values = add_values(candidate_sqls, db_context, td)
            except:
                candidates_with_values = candidate_sqls
                
            if mode == 'train' or mode == 'dev':
                labels = [1 if i in gold_sql_indices else 0 for i in range(candidate_num)] 
                # Shuffle the list
                c = list(zip(candidate_dialects, labels))
                random.shuffle(c)
                candidate_dialects, labels = zip(*c)
                ins = {
                    "index": index,
                    "db_id": db_id,
                    "question": question,
                    "candidates": candidate_dialects
                }
                ins["labels"] = labels
            else: 
                labels = [1 if i in gold_sql_indices else 0 for i in range(candidate_num)] 
                ins = {
                    "index": index,
                    "db_id": db_id,
                    "question": question,
                    "candidates": candidates_with_values,
                    "original_cadidates": candidate_sqls
                }
                ins["candidate_sqls"] = candidates_with_values
                ins["labels"] = labels
            
            output.append(ins)

    print(f"total data: {total_count}")
    print(f"output length: {len(output)}")
    if debug: 
        print(f"hit generation count: {hit_gen_num} miss generation count: {miss_gen_num}")
        checker.print_candidategen_total_result(hit_gen_num, candidate_num)
        checker.export_candidategen_miss_sqls(dataset_name, model_name)

    with open(output_file.format(dataset_name), 'w') as outfile:
        json.dump(output, outfile, indent=4)
        
    return

if __name__ == "__main__":
    main()
    
