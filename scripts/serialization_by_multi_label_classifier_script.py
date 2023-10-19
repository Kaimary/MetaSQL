import os 
import click
import json
import sqlite3

from utils.spider_utils.utils import is_consistent_schema_database
from utils.sql_utils import fix_missing_join_condition
from configs.config import DIR_PATH, SERIALIZE_DATA_DIR

@click.command()
@click.argument("test_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("model_output_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("metaor_output_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("tables_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("db_dir", type=click.Path(exists=True, dir_okay=True))
def main(test_file, model_output_file, metaor_output_file, tables_file, db_dir):
    mode = "test"
    serialization_dir = f'{DIR_PATH}{SERIALIZE_DATA_DIR}/{mode}'
    if not os.path.exists(serialization_dir): os.makedirs(serialization_dir)
    # schema = {}
    # table = {}
    # table_dict = {}
    dataset = open(test_file, 'r')
    dataset_json = json.load(dataset)
   
    metaor = open(metaor_output_file)
    outputs = [line.strip() for line in metaor.readlines()]

    pos = 0
    model_output = open(model_output_file)
    lines = model_output.readlines()
    for idx, output in enumerate(outputs):
        serialization_file = f'{serialization_dir}/{idx}.txt'
        output_lenth = len(output.split(', '))
        sqls = []
        dialects = []
        db_id = dataset_json[idx]['db_id']
        # if db_id not in schema:
        #     db_file = os.path.join(db_dir, db_id, db_id + ".sqlite")
        #     if not os.path.isfile(db_file): s = get_schema_from_json(db_id, tables_file)
        #     else: s = get_schema(db_file)
        #     _, t, td = read_single_dataset_schema(tables_file, db_id)   
        #     schema[db_id] = s
        #     table[db_id] = t
        #     table_dict[db_id] = td

        db = os.path.join(db_dir, db_id, db_id + ".sqlite")
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        # !!! There exists inconsistency on schema definition between table file and database file in Spider test set
        # Examine SQL queries only if consistency holds
        bcont = is_consistent_schema_database(db_id, db_dir, tables_file)
        for sql in lines[pos: pos+output_lenth]:
            sql = fix_missing_join_condition(sql.strip(), db_id, tables_file)
            # If join valid and not duplicate, add to the serialization
            if sql and sql not in sqls:
                try:
                    # If inconsistency exists, skip sql execution
                    if bcont:
                        cursor.execute(sql)
                    sqls.append(sql)
                    dialects.append(sql)
                except:
                    pass
        pos += output_lenth
        datafile = open(serialization_file, 'a')
        for sql, dialect in zip(sqls, dialects):
            line = f'{sql}\t{dialect}\n'
            datafile.write(line)
        datafile.close()


if __name__ == "__main__":
    main()