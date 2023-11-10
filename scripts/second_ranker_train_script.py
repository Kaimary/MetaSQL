import shutil
import sys

from allennlp.commands import main

if __name__ == '__main__':
    config_file = "allenmodels/train_configs/listwise_ranker_distributed.jsonnet"
    serialization_dir = sys.argv[1]
    # serialization_dir = 'tmp'
    # shutil.rmtree(serialization_dir, ignore_errors=True)

    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        config_file,
        "-s", serialization_dir,
        "--include-package", "allenmodels.dataset_readers.listwise_ranker_reader_distributed",
        "--include-package", "allenmodels.models.semantic_matcher.listwise_ranker_multigrained"
    ]

    main()