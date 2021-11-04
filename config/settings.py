from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / 'dataset'

TRAIN_RECORD_PATH = DATASET_PATH / 'fuji1t_bin_ver2.tfrecords'

TEST_RECORD_PATH = DATASET_PATH
