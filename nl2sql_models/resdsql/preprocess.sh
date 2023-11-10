set -e

# preprocess train_spider dataset
python preprocessing.py \
    --mode "train" \
    --table_path "./data/spider/tables.json" \
    --input_dataset_path "./data/spider/resdsql_train_spider_meta.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_train_spider.json" \
    --db_path "./database" \
    --target_type "sql"

# preprocess dev dataset
python preprocessing.py \
    --mode "eval" \
    --table_path "./data/spider/tables.json" \
    --input_dataset_path "./data/spider/dev_meta.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_dev.json" \
    --db_path "./database"\
    --target_type "sql"