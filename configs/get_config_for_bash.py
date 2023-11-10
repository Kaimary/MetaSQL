import os
import sys

dir_path = os.getcwd()

config_vars = {}
with open(dir_path + '/configs/config.py', 'r') as f:
    for line in f:
        if '=' in line:
            k, v = line.split('=', 1)
            k = k.strip()
            if k in ["RETRIEVAL_MODEL_EMBEDDING_DIMENSION", "RETRIEVAL_EMBEDDING_MODEL_NAME", \
                "RERANKER_EMBEDDING_MODEL_NAME", "RERANKER_MODEL_NAME", 'MODE', 'DEBUG']:
                config_vars[k] = v.strip().strip("'")
            elif k in ['SEMSIMILARITY_TRIPLE_DATA_GZ_FILE', 'RERANKER_TRAIN_DATA_FILE', \
                'RERANKER_DEV_DATA_FILE', 'RERANKER_MODEL_DIR', 'RETRIEVAL_MODEL_DIR']:
                config_vars[k] = dir_path + v.format('spider').strip().strip("'")
                # print(f"{idx} {k}:{config_vars[k]}")
            else:
                config_vars[k] = dir_path + v.strip().strip("'")
# print(f"config_vars:{config_vars}")
print(
    f"{config_vars['RETRIEVAL_EMBEDDING_MODEL_NAME']}@{config_vars['RERANKER_EMBEDDING_MODEL_NAME']}@"
    f"{config_vars['RERANKER_MODEL_NAME']}@{config_vars['SEMSIMILARITY_TRIPLE_DATA_GZ_FILE']}@"
    f"{config_vars['RETRIEVAL_MODEL_DIR']}@{config_vars['RERANKER_TRAIN_DATA_FILE']}@"
    f"{config_vars['RERANKER_DEV_DATA_FILE']}@{config_vars['RERANKER_MODEL_DIR']}@"
    f"{config_vars['MODE']}@{config_vars['DEBUG']}"
)
