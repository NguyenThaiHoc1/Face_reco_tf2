from FaceRecognition import FaceRec
from config import config_train
from config import settings
from dataloader.loader import DataLoader


def _training_step(iteritor_train):
    inputs, label = next(iteritor_train)
    print(label)


if __name__ == '__main__':
    loader = DataLoader(data_path_train=settings.TRAIN_RECORD_PATH, batch_size=config_train.BATCH_SIZE,
                        binary_img=config_train.BINARY_IMG)

    face_trainer = FaceRec(model=None, loader=loader, epochs=10, learning_rate=1e-5)

    face_trainer.training()
    # train_dataset = iter(loader.train)
    # count = 0
    # while count < 10:
    #     _training_step(train_dataset)
    #     count += 1

    print("Done")