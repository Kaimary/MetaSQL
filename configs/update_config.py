import json
import sys
import _jsonnet

# with open("sqlgenv2/train_configs/ranker_pair.jsonnet", 'r+') as f:
with open(sys.argv[1], 'r+') as f:
    config = json.loads(_jsonnet.evaluate_file(sys.argv[1]))
    # match = re.search(r'train_(.)+', config["train_data_path"])
    # if match is None:
    #     print(
    #         "train_data_path has invalid format:",
    #         config["train_data_path"],
    #         file=sys.stderr
    #     )
    #     sys.exit(-1)

    # new_train_data_path = re.sub(
    #     r'train_(.)+', "train_{}".format(train_file), config["train_data_path"]
    # )
    # print(config)
    config["dataset_reader"]["tokenizer"]["model_name"] = sys.argv[2]
    config["dataset_reader"]["token_indexers"]["bert"]["model_name"] = sys.argv[2]
    config["validation_dataset_reader"]["tokenizer"]["model_name"] = sys.argv[2]
    config["validation_dataset_reader"]["token_indexers"]["bert"]["model_name"] = sys.argv[2]
    config["model"]["text_field_embedder"]["token_embedders"]["bert"]["model_name"] = sys.argv[2]
    config["model"]["encoder"]["pretrained_model"] = sys.argv[2]
    config["train_data_path"] = sys.argv[3]
    config["validation_data_path"] = sys.argv[4]

    f.seek(0)  # rewind
    json.dump(config, f, indent=4)
    f.truncate()