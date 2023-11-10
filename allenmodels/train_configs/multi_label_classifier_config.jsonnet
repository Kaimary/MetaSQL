local CLASSES=24;
local model_name="roberta-large";

{
    "data_loader": {
        "batch_size": 5
    },
    "dataset_reader": {
        // "max_instances": 300,
        "token_indexers": {
            "bert": {
                "model_name": model_name,
                "type": "pretrained_transformer"
            }
        },
        "tokenizer": {
            // "add_special_tokens": false,
            "model_name": model_name,
            "type": "pretrained_transformer"
        },
        "type": "metadata_reader"
    },
    "validation_dataset_reader": {
        // "max_instances": 10,
        "token_indexers": {
            "bert": {
                "model_name": model_name,
                "type": "pretrained_transformer"
            }
        },
        "tokenizer": {
            // "add_special_tokens": false,
            "model_name": model_name,
            "type": "pretrained_transformer"
        },
        "type": "metadata_reader"
    },
    "train_data_path": "data/multi_label_classifier/train2.json",
    "validation_data_path": "data/multi_label_classifier/dev2.json",
    "model": {
        "encoder": {
            "pretrained_model": model_name,
            "type": "bert_pooler"
        },
        "text_field_embedder": {
            "token_embedders": {
                "bert": {
                    "model_name": model_name,
                    "type": "pretrained_transformer"
                }
            }
        },
        "schema_encoder": {
            "type": "relation_transformer",
            "hidden_size": 512,
            "ff_size": 512,
            "num_layers": 8,
            "tfixup": false,
            "dropout": 0.1
        },
        "classifier_feedforward":{
            "input_dim": 512,
            "num_layers": 2,
            "hidden_dims": [256, CLASSES],
            "activations": ["relu", "linear"],
            "dropout": [0.2, 0.0],
        },
        "type": "metadata_classifier"
    },
    "trainer": {
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "factor": 0.5,
            "mode": "max",
            "patience": 0,
            "type": "reduce_on_plateau"
        },
        // "learning_rate_scheduler": {
        //     "type": "polynomial_decay",
        //     "warmup_steps": 2800,
        //     "power": 0.5
        // },
        "num_epochs": 50,
        "optimizer": {
            "lr": 1e-05,
            "type": "adam"
        },
        "num_gradient_accumulation_steps": 3,
        "patience": 10,
        "validation_metric": "+f1"
    }
}
