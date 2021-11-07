import tensorflow as tf


def _transform_images(data_train, is_augment):
    if is_augment:
        raise NotImplementedError
    data_train = tf.image.resize(data_train, (128, 128))
    data_train = tf.image.random_crop(data_train, (112, 112, 3))
    data_train = tf.image.random_flip_left_right(data_train)
    data_train = tf.image.random_saturation(data_train, 0.6, 1.4)
    data_train = tf.image.random_brightness(data_train, 0.4)
    data_train = data_train / 255
    return data_train


def _transform_targets(labels_train):
    return labels_train


def _parse_tfrecord(binary_img=False, is_ccrop=False):
    def parse_tfrecord(tfrecord):
        if binary_img:
            features = {
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/filename': tf.io.FixedLenFeature([], tf.string),
                'image/source_id': tf.io.FixedLenFeature([], tf.int64),
            }

            data = tf.io.parse_single_example(tfrecord, features=features)
            data_train = tf.image.decode_jpeg(data['image/encoded'], channels=3)
        else:
            raise NotImplementedError

        # convert Labels
        labels_train = tf.cast(data['image/source_id'], tf.float32)

        # preprocessing image
        data_train = _transform_images(data_train=data_train, is_augment=is_ccrop)
        labels_train = _transform_targets(labels_train=labels_train)

        return (data_train, labels_train), labels_train

    return parse_tfrecord


class DataLoader(object):
    def __init__(self, data_path_train,
                 batch_size, binary_img, num_samples):
        self.data_path_train = data_path_train
        self.batch_size = batch_size
        self.binary_img = binary_img
        self.num_samples = num_samples

    def make_dataset(self, tfrecord_name, batchsize,
                     binary_img=False, is_ccrop=False,
                     shuffle=True, buffer_size=10240):
        """load dataset from tfrecord"""
        raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
        raw_dataset = raw_dataset.repeat()
        if shuffle:
            raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)

        dataset = raw_dataset.map(
            _parse_tfrecord(binary_img=binary_img, is_ccrop=is_ccrop),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.batch(batch_size=batchsize)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    @property
    def train(self):
        return self.make_dataset(tfrecord_name=self.data_path_train,
                                 batchsize=self.batch_size,
                                 binary_img=self.binary_img)

    @property
    def steps_per_epoch(self):
        dataset_len = self.num_samples
        steps_per_epoch = dataset_len // self.batch_size
        return steps_per_epoch