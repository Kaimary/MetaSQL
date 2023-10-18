import json
import click
from copy import deepcopy
from collections import defaultdict

from configs.config import RERANKER_MISS_FILE_NAME, PRED_SQL_FILE_NAME
from utils.sql_utils import sql_nested_query_tmp_name_convert
from spider_utils.evaluation.evaluate import Evaluator, build_foreign_key_map_from_json, rebuild_sql


@click.command()
@click.argument("tables_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("db_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("reranker_model_output_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("reranker_test_data_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("experiment_dir_name", type=click.Path(exists=True, dir_okay=True))
def main(tables_file, db_dir, reranker_model_output_file, reranker_test_data_file, experiment_dir_name):
    pred_count = 0
    preds = []
    with open(reranker_model_output_file, "r") as pred_file:
        while True:
            pos = pred_file.readline()
            if len(pos) == 0:
                break
            preds.append(pos.strip())
            pred_count += 1

    miss_total = 0
    miss_pre = 0
    miss_stage3 = 0
    total_reranker_miss = defaultdict(list)
    pred_sqls = []
    evaluator = Evaluator()
    kmaps = build_foreign_key_map_from_json(tables_file)

    with open(reranker_test_data_file, "r") as reranker_file:
        data = json.load(reranker_file)
        for i, (ex, pred) in enumerate(zip(data, preds)):
            db_id = ex['db_id']
            labels = ex['labels']
            nl = ex['question']
            candidates = ex['candidates']
            candidate_sqls = ex['candidates']
            pred_sqls.append(candidate_sqls[int(pred)])

            if 1 not in labels:
                miss_pre += 1
                miss_total += 1
                if len(total_reranker_miss[db_id]) == 0:
                    total_reranker_miss[db_id].append({"miss": 1, "stage1&2": 1, "stage3": 0})
                else:
                    total_reranker_miss[db_id][0]['miss'] += 1
                    total_reranker_miss[db_id][0]['stage1&2'] += 1
            else:
                pred_pos = int(pred)
                pred_sql = candidate_sqls[pred_pos]
                p_sql = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(pred_sql), kmaps, tables_file)

                gold_pos = labels.index(1)
                gold_sql = candidate_sqls[gold_pos]
                g_sql = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(gold_sql), kmaps, tables_file)

                if evaluator.eval_exact_match(deepcopy(p_sql), deepcopy(g_sql)) != 1:
                    miss_total += 1
                    miss_stage3 += 1
                    if len(total_reranker_miss[db_id]) == 0:
                        total_reranker_miss[db_id].append({"miss": 1, "stage1&2": 0, "stage3": 1})
                    else:
                        total_reranker_miss[db_id][0]['miss'] += 1
                        total_reranker_miss[db_id][0]['stage3'] += 1

                    gold_dialect = candidates[gold_pos]
                    pred_dialect = candidates[pred_pos]

                    cur_miss = {'nl': nl, 'gold_sql': gold_sql, 'pred_sql': pred_sql, 'gold_dialect': gold_dialect,
                                'pred_dialect': pred_dialect}

                    total_reranker_miss[db_id].append(cur_miss)

    with open(experiment_dir_name + '/' + RERANKER_MISS_FILE_NAME, 'w') as f:
        f.write(f"Total test case count:{pred_count}\n")
        f.write(f"Total miss count: {miss_total}\n")
        f.write(f"Total accuracy rate: {(pred_count - miss_total) / pred_count}\n")
        f.write(f"Stage1&2 missing count: {miss_pre}\n")
        f.write(f"Stage3 missing count: {miss_stage3}\n")
        f.write(f"Stage3 miss rate: {miss_stage3 / pred_count}\n")
        json.dump(total_reranker_miss, f, indent=4)
    with open(experiment_dir_name + '/' + PRED_SQL_FILE_NAME, 'w') as f:
        f.write('\n'.join(pred_sqls))
    

    return


if __name__ == "__main__":
    main()


