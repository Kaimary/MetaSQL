import shutil
import sys

from allennlp.commands import main

config_file = "allenmodels/train_configs/multi_label_classifier_config.jsonnet"
serialization_dir = sys.argv[1]
# serialization_dir = 'tmp'

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
# shutil.rmtree(serialization_dir, ignore_errors=True)

sys.argv = [
     "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "allenmodels.dataset_readers.multi_label_classification_reader",
    "--include-package", "allenmodels.models.multi_label_classifier"
]

main()