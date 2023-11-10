import os
dir_path = os.getcwd() 

config_vars = {}
with open(dir_path+'/configs/config.py', 'r') as f:
    for line in f:
        if '=' in line:
            k,v = line.split('=', 1)
            k = k.strip()
            if k in ["RETRIEVAL_MODEL_EMBEDDING_DIMENSION", "RETRIEVAL_EMBEDDING_MODEL_NAME", \
                "RERANKER_EMBEDDING_MODEL_NAME", "RERANKER_MODEL_NAME", "CANDIDATE_NUM", \
                "RERANKER_INPUT_FILE_NAME", "PRED_FILE_NAME", "PRED_TOPK_FILE_NAME", \
                "CANDIDATE_MISS_FILE_NAME", "SQL_MISS_FILE_NAME", "RERANKER_MISS_FILE_NAME", \
                "MODEL_TAR_GZ", "MODE", "DEBUG", "USE_ORIGINAL_PREDS", "CLASSIFIER_PREDS_FILE", "META_DICT_FILE", \
                "META_FORMAT_OUTPUT_FILE", "NL2SQL_META_PREDS_FILE", "NL2SQL_PREDS_FILE", "MODEL_BIN"]:
                config_vars[k] = v.strip().strip("'")
            elif k in ['OUTPUT_DIR_RERANKER', 'RERANKER_MODEL_DIR']:
                config_vars[k] = dir_path + v.format("spider").strip().strip("'")
            # Dir-related variables
            elif k in ['SCHEMA_CLASSIFIER_MODEL_DIR', 'CLASSIFIER_MODEL_DIR', 'RERANKER_MODEL_DIR', \
                       'NL2SQL_META_MODEL_DIR', 'NL2SQL_MODEL_DIR', 'SERIALIZE_DATA_DIR']:
                config_vars[k] = dir_path + v.strip().strip("'")
            else:
                config_vars[k] = dir_path + v.strip().strip("'")
#print(f"config_vars:{config_vars}")
print(f"{config_vars['OUTPUT_DIR_RERANKER']}@{config_vars['RETRIEVAL_EMBEDDING_MODEL_NAME']}@"
    f"{config_vars['RERANKER_MODEL_DIR']}@{config_vars['RERANKER_EMBEDDING_MODEL_NAME']}@"
    f"{config_vars['RERANKER_MODEL_NAME']}@{config_vars['RERANKER_INPUT_FILE_NAME']}@"
    f"{config_vars['PRED_FILE_NAME']}@{config_vars['RERANKER_MISS_FILE_NAME']}@{config_vars['MODEL_TAR_GZ']}@"
    f"{config_vars['PRED_TOPK_FILE_NAME']}@{config_vars['CANDIDATE_NUM']}@{config_vars['MODE']}@"
    f"{config_vars['DEBUG']}@{config_vars['CLASSIFIER_PREDS_FILE']}@{config_vars['CLASSIFIER_MODEL_DIR']}@"
    f"{config_vars['META_DICT_FILE']}@{config_vars['META_FORMAT_OUTPUT_FILE']}@{config_vars['NL2SQL_META_PREDS_FILE']}@"
    f"{config_vars['NL2SQL_PREDS_FILE']}@{config_vars['NL2SQL_META_MODEL_DIR']}@{config_vars['NL2SQL_MODEL_DIR']}@"
    f"{config_vars['MODEL_BIN']}@{config_vars['SCHEMA_CLASSIFIER_MODEL_DIR']}@{config_vars['SERIALIZE_DATA_DIR']}@{config_vars['USE_ORIGINAL_PREDS']}")
