from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

DATASET_PATH = BASE_DIR / 'prepare_dataset' / 'crawler_data' / 'data' / 'training_img_aligned' / 'lfw_rgb'

OUTPUTDIR_PATH = BASE_DIR / 'dataset' / 'lfw_dataset_rgb.tfrecords'