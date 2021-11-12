import os

import bcolz
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tqdm import tqdm

from architecture.arcface_model import ArcFaceModel
from config import settings


def l2_norm(embedings, axis=1):
    norm = np.linalg.norm(embedings, axis=axis, keepdims=True)
    output = embedings / norm
    return output


def cropped_batch(imgs):
    assert len(imgs.shape) == 4
    resized_imgs = np.array([cv2.resize(img, (128, 128)) for img in imgs])
    ccropped_imgs = resized_imgs[:, 8:-8, 8:-8, :]
    return ccropped_imgs


def performce_val(embedding_size, batch_size, model,
                  carray, issame, nrof_folds=10, is_ccrop=False, is_flip=True):
    embeddings = np.zeros([len(carray), embedding_size])

    for idx in tqdm(range(0, len(carray), batch_size)):
        batch = carray[idx:idx + batch_size]
        batch = np.transpose(batch, [0, 2, 3, 1]) * 0.5 + 0.5
        batch = batch[:, :, :, ::-1]  # BGR to RGB

        if is_ccrop:
            cropped_batch(imgs=batch)

        if is_flip:
            raise NotImplementedError
        else:
            embeding_batch = model(batch)
            embeddings[idx:idx + batch_size] = l2_norm(embeding_batch)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)

    return accuracy.mean(), best_thresholds.mean()


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)

        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(threshold=thresholds[best_threshold_index],
                                                      dist=dist[test_set], actual_issame=actual_issame[test_set])
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def evaluate(embeddings, actual_issame, nrof_folds=10):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
                                                        nrof_folds=nrof_folds)

    return tpr, fpr, accuracy, best_thresholds


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
    ckpt_path = tf.train.latest_checkpoint(settings.CHECKPOINT_PATH)
    if ckpt_path is not None:
        print('[*] load ckpt from {}.'.format(ckpt_path))
        model.load_weights(ckpt_path)

    performce_val(embedding_size=settings.EMBEDDING_SHAPE, batch_size=settings.BATCH_SIZE,
                  model=model, carray=lfw, issame=lfw_issame, is_ccrop=False, is_flip=False)
