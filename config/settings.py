from pathlib import Path

# Setting PATH
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / 'dataset'

TRAIN_RECORD_PATH = DATASET_PATH / 'fuji1t_bin_ver2.tfrecords'

TEST_RECORD_PATH = DATASET_PATH / ''

CHECKPOINT_PATH = BASE_DIR / 'checkpoint'

LOGS_PATH = BASE_DIR / 'logs'

# Training
INPUT_SIZE = 112
CHANNELS = 3
TYPE_HEAD = 'ArcHead'
TYPE_BACKBONE = 'ResNet50'
MARGIN = 0.5
LOGIST_SCALE = 64
EMBEDDING_SHAPE = 512
NUM_CLASSES = 2
NUM_SAMPLES = 35

# Hyperparamater training
W_DECAY = 5e-4
EPOCHS = 10
LEARNING_RATE = 1e-5

# DATALOADER
BATCH_SIZE = 32
BINARY_IMG = True
