import time

import cv2
import numpy as np
from absl import app, flags
from absl.flags import FLAGS

from config import settings
from dataloader.loader import DataLoader

flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_boolean('binary_img', True, 'whether use binary file or not')
flags.DEFINE_boolean('is_ccrop', True, 'whether use central cropping or not')
flags.DEFINE_boolean('visualization', True, 'whether visualize dataset or not')


def main(_):
    train_dataset = DataLoader(data_path_train=settings.TRAIN_RECORD_PATH,
                               data_path_test=settings.TRAIN_RECORD_PATH,
                               batch_size=settings.BATCH_SIZE,
                               binary_img=settings.BINARY_IMG,
                               num_samples=settings.NUM_SAMPLES)

    num_samples = 100
    start_time = time.time()
    for idx, parsed_record in enumerate(train_dataset.train.take(num_samples)):
        (x_train, _), y_train = parsed_record
        print("{} x_train: {}, y_train: {}".format(
            idx, x_train.shape, y_train.shape))

        if FLAGS.visualization:
            recon_img = np.array(x_train[0].numpy() * 255, 'uint8')
            cv2.imshow('img', cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(0) == 113:
                exit()

    print("data fps: {:.2f}".format(num_samples / (time.time() - start_time)))


if __name__ == '__main__':
    app.run(main)
