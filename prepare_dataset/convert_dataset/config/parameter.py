from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

DATASET_PATH = BASE_DIR / 'prepare_dataset' / 'crawler_data' / 'data' / 'training_img_aligned' / 'general'

OUTPUTDIR_PATH = BASE_DIR / 'dataset' / 'fuji1t_bin.tfrecords'