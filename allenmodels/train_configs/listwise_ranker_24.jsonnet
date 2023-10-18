local max_instances = 9999;
local model_name = "roberta-large";

{
    "data_loader": {
        "batch_size": 2,
        "type": "simple"
    },
    "dataset_reader": {
        "max_instances": max_instances,
        "max_tokens": 128,
        "token_indexers": {
            "bert": {
                "model_name": model_name,
                "type": "pretrained_transformer"
            }
        },
        "tokenizer": {
            // "add_special_tokens": false,
            "model_name": model_name,
            "type": "pretrained_transformer",
            "tokenizer_kwargs": {
                "additional_special_tokens": [
                    "SELECT", "JOIN", "GROUP", "BY", "ORDER", "WHERE", "INTERSECT", "UNION", "EXCEPT", "FROM", "LIMIT",
                    "COUNT(", "SUM(", "AVG(", "MAX(", "MIN(", "*",
                    "[DUPLICATE]", "[FK]", "[MISSING]", "[CONFLICT]", "[MATCH]", "[INACCURATE]", "[UNUSED]"
                ]
            }
        },
        "type": "listwise_pair_ranker_reader"
    },
    "validation_dataset_reader": {
        "max_instances": max_instances,
        "max_tokens": 128,
        "token_indexers": {
            "bert": {
                "model_name": model_name,
                "type": "pretrained_transformer"
            }
        },
        "tokenizer": {
            // "add_special_tokens": false,
            "model_name": model_name,
            "type": "pretrained_transformer",
            "tokenizer_kwargs": {
                "additional_special_tokens": [
                    "SELECT", "JOIN", "GROUP", "BY", "ORDER", "WHERE", "INTERSECT", "UNION", "EXCEPT", "FROM", "LIMIT",
                    "COUNT(", "SUM(", "AVG(", "MAX(", "MIN(", "*",
                    "[DUPLICATE]", "[FK]", "[MISSING]", "[CONFLICT]", "[MATCH]", "[INACCURATE]", "[UNUSED]"
                ]
            }
        },
        "type": "listwise_pair_ranker_reader"
    },
    "train_data_path": "data/train_with_values_1224.json",
    //"train_data_path": "data/test_test.json",
    "validation_data_path": "data/dev_with_values_1224.json",
    "model": {
        "ff_dim": 256,
        "embedder": {
            "token_embedders": {
                "bert": {
                    "model_name": model_name,
                    "type": "pretrained_transformer",
                    "tokenizer_kwargs": {
                        "additional_special_tokens": [
                            "SELECT", "JOIN", "GROUP", "BY", "ORDER", "WHERE", "INTERSECT", "UNION", "EXCEPT", "FROM", "LIMIT",
                            "COUNT(", "SUM(", "AVG(", "MAX(", "MIN(", "*",
                            "[DUPLICATE]", "[FK]", "[MISSING]", "[CONFLICT]", "[MATCH]", "[INACCURATE]", "[UNUSED]"
                        ]
                    }
                }
            }
        },
        "embedder1": {
            "token_embedders": {
                "bert": {
                    "model_name": model_name,
                    "type": "pretrained_transformer",
                    "tokenizer_kwargs": {
                        "additional_special_tokens": [
                            "SELECT", "JOIN", "GROUP", "BY", "ORDER", "WHERE", "INTERSECT", "UNION", "EXCEPT", "FROM", "LIMIT",
                            "COUNT(", "SUM(", "AVG(", "MAX(", "MIN(", "*",
                            "[DUPLICATE]", "[FK]", "[MISSING]", "[CONFLICT]", "[MATCH]", "[INACCURATE]", "[UNUSED]"
                        ]
                    }
                }
            }
        },
        "dropout": 0.2,
        "loss_weight": [1, 1],
        "triplet_loss_weight": [1, 1, 1, 1],
        "triplet_score_weight": [1, 0.3, 0.1],
        "type": "listwise_ranker"
    },
    "trainer": {
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "polynomial_decay",
            "warmup_steps": 3000,
            "power": 0.5
        },
        "num_epochs": 50,
        "optimizer": {
            "lr": 1e-05,
            "type": "adam",
           //"weight_decay": 0.01
        },
        "num_gradient_accumulation_steps": 2,
        "patience": 10,  # TODO: Fix this to
        "validation_metric": "+ndcg"
    }
}