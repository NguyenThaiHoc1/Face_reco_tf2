from config import settings
from FaceRecognition import FaceRec
from dataloader.loader import DataLoader
from architecture.arcface_model import ArcFaceModel

if __name__ == '__main__':
    loader = DataLoader(data_path_train=settings.TRAIN_RECORD_PATH,
                        batch_size=settings.BATCH_SIZE,
                        binary_img=settings.BINARY_IMG)

    model = ArcFaceModel(size=settings.INPUT_SIZE, channels=settings.CHANNELS, num_classes=settings.NUM_CLASSES,
                         margin=settings.MARGIN, logist_scale=settings.LOGIST_SCALE,
                         embd_shape=settings.EMBEDDING_SHAPE, w_decay=settings.W_DECAY,
                         head_type='ArcHead', backbone_type='ResNet50', use_pretrain=False,
                         training=True, name='arcface_model')

    face_trainer = FaceRec(model=model, loader=loader,
                           epochs=settings.EPOCHS, learning_rate=settings.LEARNING_RATE)

    face_trainer.training()
    print("Done")
