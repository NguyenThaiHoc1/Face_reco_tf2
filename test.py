import os

import bcolz
import numpy as np
import tensorflow as tf

from architecture.arcface_model import ArcFaceModel
from config import settings
from utls.utls import performce_val


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))
    return carray, issame


if __name__ == '__main__':
    data_path = './dataset/'
    lfw, lfw_issame = get_val_pair(data_path, 'lfw_align_112/lfw')

    # DONE
    model = ArcFaceModel(size=settings.INPUT_SIZE, channels=settings.CHANNELS,
                         num_classes=settings.NUM_CLASSES,
                         margin=settings.MARGIN, logist_scale=settings.LOGIST_SCALE,
                         embd_shape=settings.EMBEDDING_SHAPE, w_decay=settings.W_DECAY,
                         head_type='ArcHead', backbone_type='ResNet50', use_pretrain=False,
                         training=False, name='arcface_model')

    # Load checkpoint
    ckpt_path = tf.train.latest_checkpoint(str(settings.CHECKPOINT_PATH / 'arc_res50'))
    if ckpt_path is not None:
        print('[*] load ckpt from {}.'.format(ckpt_path))
        model.load_weights(ckpt_path)

    acc_lfw, best_th = performce_val(embedding_size=settings.EMBEDDING_SHAPE,
                                     batch_size=settings.BATCH_SIZE,
                                     model=model, carray=lfw, issame=lfw_issame,
                                     is_ccrop=False, is_flip=False)

    print("    acc {:.4f}, th: {:.2f}".format(acc_lfw, best_th))
