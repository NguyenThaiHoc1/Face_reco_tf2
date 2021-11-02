class FaceRec(object):
    def __init__(self, model):
        self.EPOCHS = 50
        self.BATCH_SIZE = 32
        self.NUMBER_OF_TRAINING_IMAGES = 320
        self.NUMBER_OF_TESTING_IMAGES = 196
        self.IMAGE_HEIGHT = 224
        self.IMAGE_WIDTH = 224

        self.model = model


