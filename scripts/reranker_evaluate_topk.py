import json
import click
from copy import deepcopy
from collections import defaultdict

from configs.config import RERANKER_MISS_TOPK_FILE_NAME, PRED_SQL_TOPK_FILE_NAME
from utils.sql_utils import sql_nested_query_tmp_name_convert
from utils.spider_utils.evaluation.evaluate import Evaluator, build_foreign_key_map_from_json, rebuild_sql


@click.command()
@click.argument("tables_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("db_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("reranker_model_output_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("reranker_input_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("experiment_dir_name", type=click.Path(exists=True, dir_okay=True))
def main(tables_file, db_dir, reranker_model_output_file, reranker_input_file, experiment_dir_name):
    pred_count = 0
    preds = []
    # Read the predicted results (top-k)
    with open(reranker_model_output_file, "r") as pred_file:
        while True:
            pos = pred_file.readline()
            if len(pos) == 0: break
            preds.append(pos.strip())
            pred_count += 1

    miss_total = 0
    miss_pre = 0
    miss_stage3 = 0
    total_reranker_miss = defaultdict(list)
    total_pred_sqls = []
    evaluator = Evaluator()
    kmaps = build_foreign_key_map_from_json(tables_file)

    with open(reranker_input_file, "r") as input:
        data = json.load(input)
        for i, (ex, pred) in enumerate(zip(data, preds)):
            db_id = ex['db_id']
            labels = ex['labels']
            nl = ex['question']
            candidates = ex['candidates']
            candidate_sqls = ex['candidate_sqls']
            top_sqls = []
            top_dialects = []
            top_sqls_index = pred.split(' ')
            for index in top_sqls_index:
                top_sqls.append(candidate_sqls[int(index)])
                top_dialects.append(candidates[int(index)])
            total_pred_sqls.append('\n'.join(top_sqls))

            if 1 not in labels:
                miss_pre += 1
                miss_total += 1
                if len(total_reranker_miss[db_id]) == 0:
                    total_reranker_miss[db_id].append({"miss": 1, "stage1&2": 1, "stage3": 0})
                else:
                    total_reranker_miss[db_id][0]['miss'] += 1
                    total_reranker_miss[db_id][0]['stage1&2'] += 1
            else:
                flag = False
                match_pos = 0
                gold_pos = labels.index(1)
                gold_sql = candidate_sqls[gold_pos]
                g_sql = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(gold_sql), kmaps, tables_file)

                for i, pred_sql in enumerate(top_sqls):
                    p_sql = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(pred_sql), kmaps, tables_file)

                    if evaluator.eval_exact_match(deepcopy(p_sql), deepcopy(g_sql)) == 1:
                        flag = True
                        match_pos = i
                        break

                if not flag:
                    miss_total += 1
                    miss_stage3 += 1
                    if len(total_reranker_miss[db_id]) == 0:
                        total_reranker_miss[db_id].append({"miss": 1, "stage1&2": 0, "stage3": 1})
                    else:
                        total_reranker_miss[db_id][0]['miss'] += 1
                        total_reranker_miss[db_id][0]['stage3'] += 1

                    gold_dialect = candidates[gold_pos]
                    cur_miss = {}
                    cur_miss['nl'] = nl
                    cur_miss['gold_sql'] = gold_sql
                    cur_miss['gold_dialect'] = gold_dialect
                    cur_miss['pred_sql'] = top_sqls
                    cur_miss['pred_dialect'] = top_dialects
                    total_reranker_miss[db_id].append(cur_miss)

                else:
                    if len(total_reranker_miss[db_id]) == 0:
                        total_reranker_miss[db_id].append({"miss": 0, "stage1&2": 0, "stage3": 0})

                    if match_pos > 0:
                        gold_dialect = candidates[gold_pos]
                        cur_miss = {}
                        cur_miss['nl'] = nl
                        cur_miss['gold_sql'] = gold_sql
                        cur_miss['gold_dialect'] = gold_dialect
                        cur_miss['match_pos'] = match_pos + 1
                        cur_miss['pred_sql'] = top_sqls[0:match_pos + 1]
                        cur_miss['pred_dialect'] = top_dialects[0:match_pos + 1]
                        total_reranker_miss[db_id].append(cur_miss)

    with open(experiment_dir_name + '/' + RERANKER_MISS_TOPK_FILE_NAME, 'w') as f:
        f.write(f"Total test case count:{pred_count}\n")
        f.write(f"Total miss count: {miss_total}\n")
        f.write(f"Total accuracy rate: {(pred_count - miss_total) / pred_count}\n")
        f.write(f"Stage1&2 missing count: {miss_pre}\n")
        f.write(f"Stage3 missing count: {miss_stage3}\n")
        f.write(f"Candidate generation miss rate: {miss_stage3 / pred_count}\n")
        json.dump(total_reranker_miss, f, indent=4)
    with open(experiment_dir_name + '/' + PRED_SQL_TOPK_FILE_NAME, 'w') as f:
        f.write('\n\n'.join(total_pred_sqls))
    return


if __name__ == "__main__":
    main()
    # main(
    #     'datasets/spider/tables.json', 'datasets/spider/database', 
    #     'output/spider/reranker/pred_topk.txt', 'output/spider/reranker/test.json',
    #     'output/spider/reranker')
