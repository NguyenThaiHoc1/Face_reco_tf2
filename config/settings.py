from pathlib import Path

# Setting PATH
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / 'dataset'

TRAIN_RECORD_PATH = DATASET_PATH / 'lfw_dataset_rgb.tfrecords'

TEST_RECORD_PATH = DATASET_PATH / ''

CHECKPOINT_PATH = BASE_DIR / 'checkpoint'

LOGS_PATH = BASE_DIR / 'logs'

# Training
INPUT_SIZE = 112
CHANNELS = 3
TYPE_HEAD = 'ArcHead'
TYPE_BACKBONE = 'ResNet50'
SUB_NAME = 'arc_res50'
MARGIN = 0.5
LOGIST_SCALE = 64
EMBEDDING_SHAPE = 512

NUM_CLASSES = 5749
NUM_SAMPLES = 14695

# Hyperparamater training
W_DECAY = 5e-4
EPOCHS = 100
LEARNING_RATE = 0.01

# DATALOADER
BATCH_SIZE = 32
BINARY_IMG = True


# -------- test algin

ONET_PATH = BASE_DIR / 'utls' / 'faceevoLVe' / 'onet.npy'
PNET_PATH = BASE_DIR / 'utls' / 'faceevoLVe' / 'pnet.npy'
RNET_PATH = BASE_DIR / 'utls' / 'faceevoLVe' / 'rnet.npy'

# PAIR PATH
TEST_PAIR_PATH = BASE_DIR / 'prepare_dataset' / 'crawler_data' / 'data' / 'validate' / 'lfw' / 'lfw_standard.txt'
DATA_DIR = BASE_DIR / 'prepare_dataset' / 'crawler_data' / 'data' / 'training_img_aligned' / 'lfw_rgb'
