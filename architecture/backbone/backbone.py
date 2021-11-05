import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet50
)


def Backbone(backbone_type, use_pretrain):
    """
    Backbone Model
    :param backbone_type: Backbone type for feature extractor
    :param user_pretrain: Using for featuring extractor
    :return: tf.Model
    """
    weights = None
    if use_pretrain:
        weights = 'imagenet'

    def _backbone(x_in):
        if backbone_type == 'ResNet50':
            return ResNet50(input_shape=x_in.shape[1:], include_top=False, weights=weights)(x_in)
        else:
            raise TypeError('backbone_type Error')

    return _backbone
