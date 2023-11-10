import os

# DIR_PATH=os.path.dirname(os.path.realpath(__file__))
DIR_PATH=os.getcwd() 
# Retrieval mdel
RETRIEVAL_MODEL_EMBEDDING_DIMENSION=768
RETRIEVAL_EMBEDDING_MODEL_NAME='all-mpnet-base-v2-sql-score10-bs15-lr5e6-lgesql0802'
SEMSIMILARITY_TRIPLE_DATA_GZ_FILE='/output/{0}/first-stage-ranker/sentence_emb.tsv.gz'
SEMSIMILARITY_TRIPLE_DATA_FINETUNE_GZ_FILE='/output/spider/first-stage-ranker/spider_sentence_emb_finetune.tsv.gz'
RETRIEVAL_MODEL_DIR='/saved_models/first-stage-ranker'
CANDIDATE_NUM=10
# Default lr is 2e-5 
RETRIVIAL_MODEL_LEARNING_RATE=5e-6
# Re-ranker
RERANKER_EMBEDDING_MODEL_NAME='roberta-large-lgesql0104-512-cand10'
RERANKER_MODEL_NAME='bertpooler'
RERANKER_TRAIN_DATA_FILE='/output/{0}/second-stage-ranker/reranker_train.json'
RERANKER_DEV_DATA_FILE='/output/{0}/second-stage-ranker/reranker_dev.json'
RERANKER_MODEL_DIR='/saved_models/second-stage-ranker'
RERANKER_TEST_DATA_FILE='/output/{0}/second-stage-ranker/reranker_test.json'
RERANKER_MODEL_OUTPUT_FILE='/output/spider/second-stage-ranker/output.txt'
# Synthesis serialization
SERIALIZE_DATA_DIR='/output/spider/serialization'
# Output
PRED_FILE='/output/spider/pred.txt'
OUTPUT_DIR_RERANKER='/output/{0}'
RERANKER_INPUT_FILE_NAME='test.json'
PRED_FILE_NAME='pred.txt'
PRED_SQL_FILE_NAME='pred_sql.txt'
PRED_TOPK_FILE_NAME='pred_topk.txt'
PRED_SQL_TOPK_FILE_NAME='pred_sql_topk.txt'
CANDIDATE_MISS_FILE_NAME='candidategen_miss.txt'
SQL_MISS_FILE_NAME='sqlgen_miss.txt'
RERANKER_MISS_FILE_NAME='reranker_miss.txt'
RERANKER_MISS_TOPK_FILE_NAME='reranker_miss_topk.txt'
VALUE_FILTERED_TOPK_FILE_NAME='value_filtered_miss_topk.txt'
# Spider Submission
META_DICT_FILE='meta_dict.txt'
CLASSIFIER_PREDS_FILE='output/cls_preds.txt'
CLASSIFIER_MODEL_DIR='/saved_models/multi_label_classifier'
SCHEMA_CLASSIFIER_MODEL_DIR='/saved_models/text2sql_schema_item_classifier'
META_FORMAT_OUTPUT_FILE='output/metadata.txt'
NL2SQL_META_MODEL_DIR='/saved_models/nl2sql_models/%s/meta'
NL2SQL_META_PREDS_FILE='output/meta_preds.txt'
NL2SQL_MODEL_DIR='/saved_models/nl2sql_models/%s/base'
NL2SQL_PREDS_FILE='output/origin_preds.txt'
# Misc.
MODEL_TAR_GZ='model.tar.gz'
MODEL_BIN='model.bin'
MODE='test'
DEBUG=False
USE_ORIGINAL_PREDS=False
TOP_NUM=5