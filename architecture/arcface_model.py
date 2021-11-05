from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from architecture.backbone.backbone import Backbone
from architecture.other.other_func import (
    ArcHead, NormHead
)
from architecture.other.other_func import OutputLayer


def ArcFaceModel(size=None, channels=3, num_classes=None, name='arcface_model',
                 margin=0.5, logist_scale=64, embd_shape=512,
                 head_type='ArcHead', backbone_type='ResNet50',
                 w_decay=5e-4, use_pretrain=True, training=False):
    """Arc Face Model"""
    x = inputs = Input([size, size, channels], name='input_image')

    x = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain)(x)

    embds = OutputLayer(embd_shape, w_decay=w_decay)(x)

    if training:
        assert num_classes is not None
        labels = Input([], name='label')
        if head_type == 'ArcHead':
            logist = ArcHead(num_classes=num_classes, margin=margin, logist_scale=logist_scale)(embds, labels)
        else:
            logist = NormHead(num_classes=num_classes, w_decay=w_decay)(embds)
        return Model((inputs, labels), logist, name=name)
    else:
        return Model(inputs, embds, name=name)


if __name__ == '__main__':
    """Testing model"""
    from config import settings
    model = ArcFaceModel(size=settings.INPUT_SIZE, channels=settings.CHANNELS, num_classes=settings.NUM_CLASSES,
                         margin=settings.MARGIN, logist_scale=settings.LOGIST_SCALE,
                         embd_shape=settings.EMBEDDING_SHAPE, w_decay=settings.W_DECAY,
                         head_type='ArcHead', backbone_type='ResNet50', use_pretrain=False,
                         training=True, name='arcface_model')
    model.summary()
