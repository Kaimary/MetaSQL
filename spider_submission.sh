#!/bin/bash

[ -z "$1" ] && echo "First argument is the NL2SQL model name." && exit 1
MODEL_NAME="$1"

[ -z "$2" ] && echo "First argument is the test json file." && exit 1
TEST_FILE="$2"

[ -z "$3" ] && echo "Second argument is the schema file." && exit 1
TABLES_FILE="$3"

[ -z "$4" ] && echo "Third argument is the directory of the databases." && exit 1
DB_DIR="$4"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "===================================================================================================================================="
echo "INFO     ****** MetaSQL Testing Pipeline Start ******"
echo "$TABLES_FILE" > params.txt
echo "$DB_DIR" >> params.txt

output=`python3 -m configs.get_config_for_test_bash`
OUTPUT_DIR_RERANKER=$(cut -d'@' -f1 <<< "$output")
RETRIEVAL_EMBEDDING_MODEL_NAME=$(cut -d'@' -f2 <<< "$output")
RERANKER_MODEL_DIR=$(cut -d'@' -f3 <<< "$output")
RERANKER_EMBEDDING_MODEL_NAME=$(cut -d'@' -f4 <<< "$output")
RERANKER_MODEL_NAME=$(cut -d'@' -f5 <<< "$output")
RERANKER_INPUT_FILE_NAME=$(cut -d'@' -f6 <<< "$output")
PRED_FILE_NAME=$(cut -d'@' -f7 <<< "$output")
RERANKER_MISS_FILE_NAME=$(cut -d'@' -f8 <<< "$output")
MODEL_TAR_GZ=$(cut -d'@' -f9 <<< "$output")
PRED_TOPK_FILE_NAME=$(cut -d'@' -f10 <<< "$output")
CANDIDATE_NUM=$(cut -d'@' -f11 <<< "$output")
MODE=$(cut -d'@' -f12 <<< "$output")
DEBUG=$(cut -d'@' -f13 <<< "$output")
CLASSIFIER_PREDS_FILE=$(cut -d'@' -f14 <<< "$output")
CLASSIFIER_MODEL_DIR=$(cut -d'@' -f15 <<< "$output")
META_DICT_FILE=$(cut -d'@' -f16 <<< "$output")
META_FORMAT_OUTPUT_FILE=$(cut -d'@' -f17 <<< "$output")
NL2SQL_META_PREDS_FILE=$(cut -d'@' -f18 <<< "$output")
NL2SQL_PREDS_FILE=$(cut -d'@' -f19 <<< "$output")
NL2SQL_META_MODEL_DIR=$(cut -d'@' -f20 <<< "$output")
NL2SQL_MODEL_DIR=$(cut -d'@' -f21 <<< "$output")
MODEL_BIN=$(cut -d'@' -f22 <<< "$output")
SCHEMA_CLASSIFIER_MODEL_DIR=$(cut -d'@' -f23 <<< "$output")
SERIALIZE_DATA_DIR=$(cut -d'@' -f24 <<< "$output")

printf -v NL2SQL_MODEL_DIR "$NL2SQL_MODEL_DIR" $MODEL_NAME
printf -v NL2SQL_META_MODEL_DIR "$NL2SQL_META_MODEL_DIR" $MODEL_NAME

DATASET_NAME="spider"
EXPERIMENT_DIR_NAME=$OUTPUT_DIR_RERANKER/$DATASET_NAME\_$MODEL_NAME\_$CANDIDATE_NUM\_$RETRIEVAL_EMBEDDING_MODEL_NAME\_$RERANKER_EMBEDDING_MODEL_NAME\_$RERANKER_MODEL_NAME
if [ ! -d $EXPERIMENT_DIR_NAME ]; then
    mkdir -p $EXPERIMENT_DIR_NAME
fi
RERANKER_INPUT_FILE=$EXPERIMENT_DIR_NAME/$RERANKER_INPUT_FILE_NAME
RERANKER_MODEL_FILE=$RERANKER_MODEL_DIR/$RERANKER_MODEL_NAME\_$RERANKER_EMBEDDING_MODEL_NAME/$MODEL_TAR_GZ
RERANKER_MODEL_OUTPUT_FILE=$EXPERIMENT_DIR_NAME/$PRED_FILE_NAME
RERANKER_MODEL_OUTPUT_TOPK_FILE=$EXPERIMENT_DIR_NAME/$PRED_TOPK_FILE_NAME
RERANKER_MODEL_OUTPUT_SQL_FILE=${RERANKER_MODEL_OUTPUT_FILE/.txt/_sql.txt}
RERANKER_MODEL_OUTPUT_TOPK_SQL_FILE=${RERANKER_MODEL_OUTPUT_FILE/.txt/_sql_topk.txt}
EVALUATE_OUTPUT_FILE=${RERANKER_MODEL_OUTPUT_FILE/.txt/_evaluate.txt}
VALUE_FILTERED_OUTPUT_SQL_FILE=${RERANKER_MODEL_OUTPUT_FILE/.txt/_sql_value_filtered.txt}
VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE=${RERANKER_MODEL_OUTPUT_FILE/.txt/_sql_topk_value_filtered.txt}
CLASSIFIER_MODEL_FILE=$CLASSIFIER_MODEL_DIR/$MODEL_BIN
LGESQL_META_MODEL_FILE=$NL2SQL_META_MODEL_DIR/$MODEL_BIN
LGESQL_MODEL_FILE=$NL2SQL_MODEL_DIR/$MODEL_BIN

# Get predictions from multi-label classification model
echo "INFO     [Stage 1] Multi-label classification model inferencing ......"
if [ -f $CLASSIFIER_MODEL_FILE -a ! -f $CLASSIFIER_PREDS_FILE ]; then
    python3 -m scripts.infer_multi_label_script --db_dir "$DB_DIR" --table_path "$TABLES_FILE" \
    --dataset_path "$TEST_FILE" --saved_model "$CLASSIFIER_MODEL_DIR" --output_path "$CLASSIFIER_PREDS_FILE" --use_gpu || exit $?
    python3 -m scripts.multi_label_classifier_output_format_script $CLASSIFIER_PREDS_FILE \
    $META_DICT_FILE $META_FORMAT_OUTPUT_FILE || exit $?
    echo "INFO     Multi-label classification model inference complete & outputs formatting done!"
    echo "===================================================================================================================================="
else
    echo "WARNING     \`$CLASSIFIER_MODEL_FILE\` not exists or \`$CLASSIFIER_PREDS_FILE\` exists."
fi

# Get the predictions from the LGESQL+Meta model
echo "INFO     [Stage 2-(1)] NL2SQL+meta model inferencing ......"
if [ ! -f $NL2SQL_META_PREDS_FILE ]; then
    case $MODEL_NAME in
        "lgesql")
        python3 -m nl2sql_models.lgesql.infer_with_meta_script --metadata_dict_path "$META_DICT_FILE" \
        --metadata_output_path "$META_FORMAT_OUTPUT_FILE" --db_dir "$DB_DIR" --table_path "$TABLES_FILE" \
        --saved_table_path "" --dataset_path "$TEST_FILE" --saved_model "$NL2SQL_META_MODEL_DIR" \
        --output_path "$NL2SQL_META_PREDS_FILE" || exit $?
        ;;
        "resdsql")
        echo "resdsql"
        ;;
        "gap")
        echo "gap"
        ;;
        "bridge")
        echo "bridge"
        ;;
        *)
        echo "unknown NL2SQL model!"
        exit;
        ;;
    esac
    python3 -m scripts.check_empty_line_script $NL2SQL_META_PREDS_FILE || exit $?
    echo "INFO     NL2SQL+meta model inference complete!"
    echo "===================================================================================================================================="
else
    echo "WARNING     \`$LGESQL_META_MODEL_FILE\` not exists or \`$NL2SQL_META_PREDS_FILE\` exists."
fi

# Get the predictions from the original LGESQL model
echo "INFO     [Stage 2-(2)] NL2SQL model inferencing ......"
if [ ! -f $NL2SQL_PREDS_FILE ]; then
    case $MODEL_NAME in
        "lgesql")
        python3 -m nl2sql_models.lgesql.infer_script --db_dir "$DB_DIR" --table_path "$TABLES_FILE" \
        --saved_table_path "output/tables.bin" --dataset_path "$TEST_FILE" \
        --saved_model "$NL2SQL_MODEL_DIR" --output_path "$NL2SQL_PREDS_FILE" || exit $?
        ;;
        "resdsql")
        echo "resdsql"
                ;;
        "gap")
        echo "gap"
        ;;
        "bridge")
        echo "bridge"
        ;;
        *)
        echo "unknown NL2SQL model!"
        exit;
        ;;
    esac
    python3 -m scripts.check_empty_line_script $NL2SQL_PREDS_FILE || exit $?
    echo "INFO     NL2SQL model inference complete!"
    echo "===================================================================================================================================="
else
    echo "WARNING     \`$NL2SQL_MODEL_FILE\` not exists or \`$NL2SQL_PREDS_FILE\` exists."
fi

echo "INFO     [Stage 2-(3)] Outputs Serialization ......"
if [ ! -d $SERIALIZE_DATA_DIR ]; then
    python3 -m scripts.serialization_by_multi_label_classifier_script  $TEST_FILE $NL2SQL_META_PREDS_FILE \
    $META_FORMAT_OUTPUT_FILE $TABLES_FILE $DB_DIR || exit $?
    echo "INFO     serialization complete!"
    echo "===================================================================================================================================="
else
    echo "WARNING     \`$SERIALIZE_DATA_DIR\` exists."
fi

# Generate the input data for the re-ranking model
echo "INFO     [Stage 3-(1)] First-stage ranking inferencing ......"
if [ ! -f $RERANKER_INPUT_FILE ]; then
    python3 -m scripts.reranker_script $DATASET_NAME $MODEL_NAME $RETRIEVAL_EMBEDDING_MODEL_NAME \
    $TEST_FILE $NL2SQL_PREDS_FILE $TABLES_FILE $DB_DIR $CANDIDATE_NUM \
    $MODE $DEBUG $RERANKER_INPUT_FILE || exit $?
    echo "INFO     First-stage ranking inference complete & Second-stage re-ranking data done!"
    echo "===================================================================================================================================="
else
    echo "WARNING     \`$RERANKER_INPUT_FILE\` exists!"
fi

# if [ ! -f $EXPERIMENT_DIR_NAME/preprocessed_test.json ]; then
#     echo "INFO     Start to preprocessing $RERANKER_INPUT_FILE"  
#     python -m scripts.preprocessing --mode test --table_path $TABLES_FILE --input_dataset_path $RERANKER_INPUT_FILE \
#         --output_dataset_path $EXPERIMENT_DIR_NAME/preprocessed_test.json --db_path $DB_DIR --target_type sql

#     # predict probability for each schema item in the eval set
#     python -m scripts.schema_item_classifier --batch_size 32 --device 0 --seed 42 \
#         --save_path $SCHEMA_CLASSIFIER_MODEL_DIR \
#         --dev_filepath $EXPERIMENT_DIR_NAME/preprocessed_test.json \
#         --output_filepath $EXPERIMENT_DIR_NAME/test_with_probs.json \
#         --use_contents --add_fk_info --mode "test"

#     # generate text2sql development dataset
#     python -m scripts.text2sql_data_generator \
#         --input_dataset_path $EXPERIMENT_DIR_NAME/test_with_probs.json \
#         --output_dataset_path $EXPERIMENT_DIR_NAME/resdsql_test.json \
#         --topk_table_num 4 --topk_column_num 5 --mode test --use_contents \
#         --add_fk_info --output_skeleton --target_type sql
#     echo "INFO     Re-ranking model inference complete!"
#     echo "=================================================================="
# else
#     echo "$RERANKER_INPUT_FILE was already preprocessed."
#     exit;
# fi

# Inference for top-1
echo "INFO     [Stage 3-(2)] Second-stage re-ranking top-1 inferencing ......"
if [ ! -f $RERANKER_MODEL_OUTPUT_FILE ]; then
    allennlp predict "$RERANKER_MODEL_FILE" "$RERANKER_INPUT_FILE" \
    --output-file "$RERANKER_MODEL_OUTPUT_FILE" \
    --file-friendly-logging --silent --predictor listwise-ranker --use-dataset-reader --cuda-device 0 \
    --include-package allenmodels.dataset_readers.listwise_ranker_reader_distributed \
    --include-package allenmodels.models.semantic_matcher.listwise_ranker_multigrained \
    --include-package allenmodels.modules.relation_transformer \
    --include-package allenmodels.predictors.ranker_predictor || exit $?
    echo "INFO     Second-stage re-ranking inference complete!"
    echo "===================================================================================================================================="
else
    echo "WARNING     \`$RERANKER_MODEL_FILE\` not exists or \`$RERANKER_MODEL_OUTPUT_FILE\` exists."
    # exit;
fi

# Evaluate re-ranker model
echo "INFO     [Stage 3-(3) Second-stage re-ranking (top-1 results) evaluating ......"
if [ -f $RERANKER_MODEL_OUTPUT_FILE -a ! -f $RERANKER_MODEL_OUTPUT_SQL_FILE ]; then
    python3 -m scripts.reranker_evaluate $TABLES_FILE $DB_DIR $RERANKER_MODEL_OUTPUT_FILE \
    $RERANKER_INPUT_FILE $EXPERIMENT_DIR_NAME || exit $?
    echo "INFO     Second-stage re-ranking evaluation complete!"
    echo "===================================================================================================================================="
else
    echo "WARNING     \`RERANKER_MODEL_OUTPUT_FILE\` not exists or \`$RERANKER_MODEL_OUTPUT_SQL_FILE\` exists"
    # exit;
fi

# Inference for top-k
echo "INFO     [Stage 3-(4)] Second-stage re-ranking top-k inferencing ....."
if [ -f $RERANKER_MODEL_FILE -a ! -f $RERANKER_MODEL_OUTPUT_TOPK_FILE ]; then
    allennlp predict "$RERANKER_MODEL_FILE" "$RERANKER_INPUT_FILE" \
    --output-file "$RERANKER_MODEL_OUTPUT_TOPK_FILE" \
    --file-friendly-logging --silent --predictor listwise-ranker --use-dataset-reader --cuda-device 0 \
     --include-package allenmodels.dataset_readers.listwise_ranker_reader_distributed \
     --include-package allenmodels.models.semantic_matcher.listwise_ranker_multigrained \
     --include-package allenmodels.modules.relation_transformer \
     --include-package allenmodels.predictors.ranker_predictor_topk || exit $?
    echo "INFO     Second-stage re-ranking inference (top-k) complete!"
    echo "===================================================================================================================================="
else
    echo "WARNING     \`$RERANKER_MODEL_FILE\` not exists or \`$RERANKER_MODEL_OUTPUT_TOPK_FILE\` exists."
    # exit;
fi

# Evaluate for top-k
echo "INFO     [Stage 3-(5)] Second-stage re-ranking (top-k results) evaluating ......"
if [ -f $RERANKER_MODEL_OUTPUT_TOPK_FILE -a ! -f $RERANKER_MODEL_OUTPUT_TOPK_SQL_FILE ]; then
    python3 -m scripts.reranker_evaluate_topk $TABLES_FILE $DB_DIR \
    $RERANKER_MODEL_OUTPUT_TOPK_FILE $RERANKER_INPUT_FILE $EXPERIMENT_DIR_NAME || exit $?
    echo "INFO     Second-stage re-ranking evaluation complete!"
    echo "===================================================================================================================================="
else
    echo "WARNING     \`$RERANKER_MODEL_OUTPUT_TOPK_SQL_FILE\` exists"
    # exit;
fi

# Value filtered
echo "INFO     [Stage 3-(6)] Postprocessing ......"
if [ -f $RERANKER_MODEL_OUTPUT_SQL_FILE -a ! -f $VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE ]; then
    python3 -m scripts.candidate_filter_top10 "$TEST_FILE" "$NL2SQL_PREDS_FILE" "$RERANKER_INPUT_FILE" \
    "$RERANKER_MODEL_OUTPUT_TOPK_SQL_FILE" "$TABLES_FILE" "$DB_DIR" \
    "$VALUE_FILTERED_OUTPUT_SQL_FILE" "$VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE" || exit $?
    echo "INFO     Postprocessing complete!"
    echo "===================================================================================================================================="
else
    echo "WARNING     \`$VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE\` exist!"
    exit;
fi

# Final Evaluation
echo "INFO     Spider script evaluating ......"
if [ -f $VALUE_FILTERED_OUTPUT_SQL_FILE -a ! -f $EVALUATE_OUTPUT_FILE ]; then
    python3 -m utils.spider_utils.evaluation.evaluate --gold "data/dev_gold.sql" --pred "$VALUE_FILTERED_OUTPUT_SQL_FILE" \
    --etype "match" --db "$DB_DIR" --table "$TABLES_FILE" \
    --candidates "$VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE" > "$EVALUATE_OUTPUT_FILE"
    echo "Spider evaluation complete! Results are saved in \`$EVALUATE_OUTPUT_FILE\`"
    echo "===================================================================================================================================="
else
    echo "\`$EVALUATE_OUTPUT_FILE\` exist!"
    echo "===================================================================================================================================="
    # exit
fi
# echo "Test Pipeline completed!"
# rm -rf ../.cache
# rm -rf ../.allennlp
# rm -rf ../.conda
# rm -rf ../.dgl
# rm -rf ../.nv
# cp -v "$VALUE_FILTERED_OUTPUT_SQL_FILE" "$DIR"/predicted_sql.txt