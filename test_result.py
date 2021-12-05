import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import tensorflow as tf

from architecture.arcface_model import ArcFaceModel
from config import settings
from utls.evalute.standard_measures import evaluate_lfw
from utls.evalute.utlis import evalute
from utls.other import lfw

"""
    LFW:  accuracy ==> 97%
    
    vấn đề 
    - Read file bằng OPENCV 
    - Khi training ta dùng tensorflow
        read image từ tensorflow chứ không phải tensorflow nên nó bị giảm từ 99,35 ==> 97 

"""


def preprocess(np_array):
    img = cv2.resize(np_array, (settings.INPUT_SIZE, settings.INPUT_SIZE))
    img = img.astype(np.float32) / 255.
    img = np.transpose(img, [2, 0, 1])
    return img


def pipline(path_array):
    np_images = []
    for abspath_filename in path_array:
        # luu y hinh anh nay da duoc algin truoc do
        # o day chi doc ra
        np_image = cv2.imread(abspath_filename)
        # np_image = np.asarray(Image.open(abspath_filename))
        np_preprocessed = preprocess(np_image)
        np_images.append(np_preprocessed)
    return np.asarray(np_images)


if __name__ == '__main__':
    pairs = lfw.read_pairs(pairs_filename=settings.TEST_PAIR_PATH)

    paths, actual_issame = lfw.get_paths(settings.DATA_DIR, pairs)

    list_array = pipline(path_array=paths)

    model = ArcFaceModel(size=settings.INPUT_SIZE, channels=settings.CHANNELS,
                         num_classes=settings.NUM_CLASSES,
                         margin=settings.MARGIN, logist_scale=settings.LOGIST_SCALE,
                         embd_shape=settings.EMBEDDING_SHAPE, w_decay=settings.W_DECAY,
                         head_type='ArcHead', backbone_type='ResNet50', use_pretrain=False,
                         training=False, name='arcface_model')

    ckpt_path = tf.train.latest_checkpoint(str(settings.CHECKPOINT_PATH / 'arc_res50'))
    if ckpt_path is not None:
        print('[*] load ckpt from {}.'.format(ckpt_path))
        model.load_weights(ckpt_path)

    distances, labels = evalute(embedding_size=settings.EMBEDDING_SHAPE,
                                batch_size=settings.BATCH_SIZE,
                                model=model, carray=list_array, issame=actual_issame)

    metrics = evaluate_lfw(distances=distances, labels=labels)

    txt = "Accuracy on LFW: {:.4f}+-{:.4f}\nPrecision {:.4f}+-{:.4f}\nRecall {:.4f}+-{:.4f}" \
          "\nROC Area Under Curve: {:.4f}\nBest distance threshold: {:.2f}+-{:.2f}" \
          "\nTAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(
        np.mean(metrics['accuracy']),
        np.std(metrics['accuracy']),
        np.mean(metrics['precision']),
        np.std(metrics['precision']),
        np.mean(metrics['recall']),
        np.std(metrics['recall']),
        metrics['roc_auc'],
        np.mean(metrics['best_distances']),
        np.std(metrics['best_distances']),
        np.mean(metrics['tar']),
        np.std(metrics['tar']),
        np.mean(metrics['far']))
    print(txt)

    title = 'LFW metrics'
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(title, fontsize=15)
    fig.set_size_inches(14, 6)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    axes[0].set_title('distance histogram')
    sb.distplot(distances[labels == True], ax=axes[0], label='distance-true')
    sb.distplot(distances[labels == False], ax=axes[0], label='distance-false')
    axes[0].legend()

    axes[1].text(0.05, 0.3, txt, fontsize=20)
    axes[1].set_axis_off()
    plt.show()
