local debug = false;
local model_name = "roberta-large";
local batch_size_per_gpu = 2;
{
    "data_loader": {
        "batch_size": batch_size_per_gpu,
        "shuffle": true,
    },
    "dataset_reader": {
        [if debug then "max_instances"]: 10,
        "max_tokens": 128,
        "token_indexers": {
            "bert": {
                "model_name": model_name,
                "type": "pretrained_transformer"
            }
        },
        "tokenizer": {
            "model_name": model_name,
            "type": "pretrained_transformer",
            "tokenizer_kwargs": {
                "additional_special_tokens": [
                    "SELECT", "JOIN", "GROUP", "BY", "ORDER", "WHERE", "INTERSECT", "UNION", "EXCEPT", "FROM", "LIMIT",
                    "COUNT(", "SUM(", "AVG(", "MAX(", "MIN(", "*",
                    "[DUPLICATE]", "[FK]", "[MISSING]", "[CONFLICT]", "[MATCH]", "[INACCURATE]"
                ]
            }
        },
        "type": "listwise_pair_ranker_reader"
    },
    "train_data_path": if debug then "data/train_with_values_1224_test.json" else "data/train_with_values_1224.json",
    "validation_data_path": "data/dev_with_values_1224.json",
    "model": {
        "ff_dim": 256,
        "embedder": {
            "token_embedders": {
                "bert": {
                    "model_name": model_name,
                    "type": "pretrained_transformer",
                    "tokenizer_kwargs": {
                        "add_pooling_layer": false,
                        "additional_special_tokens": [
                            "SELECT", "JOIN", "GROUP", "BY", "ORDER", "WHERE", "INTERSECT", "UNION", "EXCEPT", "FROM", "LIMIT",
                            "COUNT(", "SUM(", "AVG(", "MAX(", "MIN(", "*",
                            "[DUPLICATE]", "[FK]", "[MISSING]", "[CONFLICT]", "[MATCH]", "[INACCURATE]"
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
                        "add_pooling_layer": false,
                        "additional_special_tokens": [
                            "SELECT", "JOIN", "GROUP", "BY", "ORDER", "WHERE", "INTERSECT", "UNION", "EXCEPT", "FROM", "LIMIT",
                            "COUNT(", "SUM(", "AVG(", "MAX(", "MIN(", "*",
                            "[DUPLICATE]", "[FK]", "[MISSING]", "[CONFLICT]", "[MATCH]", "[INACCURATE]"
                        ]
                    }
                }
            }
        },
        "dropout": 0.2,
        "loss_weight": [1, 1, 1],
        // "triplet_loss_weight": [1, 1, 1, 1],
        // "triplet_score_weight": [1, 0.3, 0.1],
        "type": "listwise_ranker"
    },
    "trainer": {
        "learning_rate_scheduler": {
            "type": "polynomial_decay",
            "warmup_steps": 3000,
            "power": 0.5
        },
        "num_epochs": if debug then 2 else 50,
        "optimizer": {
            "lr": 1e-05,
            "type": "adam"
        },
        // "num_gradient_accumulation_steps": 2,
        "patience": 10,  # TODO: Fix this to
        "validation_metric": "+ndcg"
    },
    "distributed": {
        "cuda_devices": if debug then [4, 5] else [4, 5, 7],
        "ddp_accelerator": {
            "type": "torch",
            "find_unused_parameters": true
        }
    },
}