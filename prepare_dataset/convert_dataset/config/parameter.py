from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

DATASET_PATH = BASE_DIR / 'prepare_dataset' / 'crawler_data' / 'data' / 'training_img_aligned' / 'lfw'

OUTPUTDIR_PATH = BASE_DIR / 'dataset' / 'lfw_dataset.tfrecords'