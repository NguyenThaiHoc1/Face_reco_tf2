from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

INPUT_PATH = BASE_DIR / 'prepare_dataset' / 'crawler_data' / 'data' / 'training_img' / 'lfw'

OUTPUT_PATH = BASE_DIR / 'prepare_dataset' / 'crawler_data' / 'data' / 'training_img_aligned' / 'lfw_rgb'
