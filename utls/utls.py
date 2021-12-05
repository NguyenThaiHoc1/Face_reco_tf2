import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tqdm import tqdm


def get_ckpt_inf(ckpt_path, steps_per_epoch):
    """get ckpt information"""
    split_list = ckpt_path.split('e_')[-1].split('_b_')
    epochs = int(split_list[0])
    batchs = int(split_list[-1].split('.ckpt')[0])
    steps = (epochs - 1) * steps_per_epoch + batchs

    return epochs, steps + 1


def write_data_tensorboard(writter, epochs, max_epochs, each_step, steps_per_epoch,
                           learning_rate, total_loss, pred_loss, reg_loss, optimizer):
    """

    :param writter:
    :param epochs:
    :param max_epochs:
    :param each_step:
    :param steps_per_epoch:
    :param learning_rate:
    :param total_loss:
    :param pred_loss:
    :param reg_loss:
    :param optimizer:
    :return:
    """
    verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}"
    print(verb_str.format(epochs, max_epochs,
                          each_step % steps_per_epoch,
                          steps_per_epoch,
                          total_loss.numpy(),
                          learning_rate.numpy()))

    with writter.as_default():
        tf.summary.scalar('loss/total loss', total_loss, step=each_step)
        tf.summary.scalar('loss/pred loss', pred_loss, step=each_step)
        tf.summary.scalar('loss/reg loss', reg_loss, step=each_step)
        tf.summary.scalar('learning rate', optimizer.lr, step=each_step)


def load_checkpoint(path_checkpoint, model, steps_per_epoch):
    ckpt_path = tf.train.latest_checkpoint(path_checkpoint)
    if ckpt_path is not None:
        print('[*] load ckpt from {}.'.format(ckpt_path))
        model.load_weights(ckpt_path)
        epochs, steps = get_ckpt_inf(ckpt_path=ckpt_path, steps_per_epoch=steps_per_epoch)
    else:
        print('[*] training from scratch.')
        epochs, steps = 1, 1
    return epochs, steps


def save_weight(model, path_dir):
    print('[*] save ckpt file!')
    model.save_weights(path_dir)


def l2_norm(embedings, axis=1):
    norm = np.linalg.norm(embedings, axis=axis, keepdims=True)
    output = embedings / norm
    return output


def cropped_batch(imgs):
    assert len(imgs.shape) == 4
    resized_imgs = np.array([cv2.resize(img, (128, 128)) for img in imgs])
    ccropped_imgs = resized_imgs[:, 8:-8, 8:-8, :]
    return ccropped_imgs


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
