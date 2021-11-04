import glob
import os
import random

import tensorflow as tf
import tqdm
from absl import app, logging

from prepare_dataset.convert_dataset.config import parameter


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Returns an float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def make_example(img_byte, id_name, real_name, filename):
    feature = {
        'image/encoded': _bytes_feature(value=img_byte),
        'image/filename': _bytes_feature(value=filename),
        'image/source_id': _int64_feature(value=id_name),
        'image/realname': _bytes_feature(value=real_name),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(_):
    dataset_path = str(parameter.DATASET_PATH)
    output_path = str(parameter.OUTPUTDIR_PATH)

    if not os.path.isdir(dataset_path):
        logging.info('Please define valid dataset path.')
    else:
        logging.info('Loading {}'.format(dataset_path))

    logging.info("Reading data list file ...")
    samples = []
    for index_name, id_name in tqdm.tqdm(enumerate(os.listdir(dataset_path))):
        img_paths = glob.glob(os.path.join(dataset_path, id_name, '*.jpg'))  # list file image JPG
        for img_path in img_paths:
            filename = os.path.join(id_name, os.path.basename(img_path))
            samples.append((img_path, index_name, id_name, filename))
    random.shuffle(samples)

    logging.info("Writting tfrecord file ...")
    with tf.io.TFRecordWriter(output_path) as writer:
        for img_path, index_name, id_name, filename in tqdm.tqdm(samples):
            tf_example = make_example(
                img_byte=open(img_path, 'rb').read(),
                id_name=int(index_name),
                real_name=str.encode(id_name),
                filename=str.encode(filename),
            )
            writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
