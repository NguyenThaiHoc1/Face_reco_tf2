import ssl

from FaceRecognition import FaceRec
from architecture.arcface_model import ArcFaceModel
from config import settings
from dataloader.loader import DataLoader
from utls.utls import load_checkpoint

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':
    # DONE
    loader = DataLoader(data_path_train=settings.TRAIN_RECORD_PATH,
                        data_path_test=settings.TRAIN_RECORD_PATH,
                        batch_size=settings.BATCH_SIZE,
                        binary_img=settings.BINARY_IMG,
                        num_samples=settings.NUM_SAMPLES)

    # DONE
    model = ArcFaceModel(size=settings.INPUT_SIZE, channels=settings.CHANNELS, num_classes=settings.NUM_CLASSES,
                         margin=settings.MARGIN, logist_scale=settings.LOGIST_SCALE,
                         embd_shape=settings.EMBEDDING_SHAPE, w_decay=settings.W_DECAY,
                         head_type='ArcHead', backbone_type='ResNet50', use_pretrain=True,
                         training=True, name='arcface_model')

    # Load checkpoint
    current_epochs, steps = load_checkpoint(path_checkpoint=settings.CHECKPOINT_PATH, model=model,
                                            steps_per_epoch=loader.steps_per_epoch)

    # NO DONE
    face_trainer = FaceRec(model=model, loader=loader,
                           current_epochs=current_epochs,
                           max_epochs=settings.EPOCHS,
                           steps=steps,
                           learning_rate=settings.LEARNING_RATE,
                           logs=str(settings.LOGS_PATH),
                           saveweight_path=str(settings.CHECKPOINT_PATH / settings.SUB_NAME))

    face_trainer.training()
    print("Done")
