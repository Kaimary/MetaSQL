{
    "data_loader": {
        "batch_size": 1,
        "type": "pytorch_dataloader"
    },
    "dataset_reader": {
        "lazy": false,
        // "max_instances": 100,
        // "max_tokens": 512,
        "token_indexers": {
            "bert": {
                "model_name": "roberta-large",
                "type": "pretrained_transformer"
            }
        },
        "token_tokenizer": {
            // "add_special_tokens": false,
            "model_name": "roberta-large",
            "type": "pretrained_transformer"
        },
        "type": "listwise_pair_ranker_reader"
    },
    "validation_dataset_reader": {
        // "max_instances": 10,
        "token_indexers": {
            "bert": {
                "model_name": "roberta-large",
                "type": "pretrained_transformer"
            }
        },
        "token_tokenizer": {
            // "add_special_tokens": false,
            "model_name": "roberta-large",
            "type": "pretrained_transformer"
        },
        "type": "listwise_pair_ranker_reader"
    },
    "train_data_path": "/home/kaimary/lfm-gap/rat-sql-gap/output/spider/reranker/reranker_train_cand10_0816.json",
    "validation_data_path": "/home/kaimary/lfm-gap/rat-sql-gap/output/spider/reranker/reranker_dev_cand10_0820.json",
    "model": {
        "dropout": 0.2,
        // "encoder": {
        //     "pretrained_model": "roberta-large",
        //     "type": "bert_pooler"
        // },
        "encoder": {
            "type": "boe",
            "embedding_dim": 256,
            "averaged": true,
        },
        "schema_encoder": {
            "type": "relation_transformer",
            "hidden_size": 512,
            "ff_size": 512,
            "num_layers": 8,
            "tfixup": false,
            "dropout": 0.1,
        },
        "text_field_embedder": {
            "token_embedders": {
                "bert": {
                    "model_name": "roberta-large",
                    "type": "pretrained_transformer"
                }
            }
        },
        "type": "listwise_pair_ranker"
    },
    "trainer": {
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "polynomial_decay",
            "warmup_steps": 5000,
            "power": 0.5
        },
        // "learning_rate_scheduler": {
        //     "factor": 0.5,
        //     "mode": "max",
        //     "patience": 0,
        //     "type": "reduce_on_plateau"
        // },
        "num_epochs": 100,
        "optimizer": {
            "lr": 1e-05,
            "type": "adam"
        },
        "num_gradient_accumulation_steps": 2,
        "patience": 10,
        "validation_metric": "+ndcg"
    }
}
