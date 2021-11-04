import tensorflow as tf

from loss.losses import SoftmaxLoss


class FaceRec(object):
    def __init__(self, model, loader, epochs, learning_rate):
        self.epochs = epochs
        self.model = model
        self.loader = loader

        self.lr = learning_rate

        # -------- setting up hyper parameter -------
        self.optimizer = None
        self.loss_fn = None
        self.writter_train = None
        self.writter_vali = None

    def _setup_optimizer(self):
        learning_rate = tf.constant(self.lr)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        return optimizer

    def _setup_loss(self):
        loss_fn = SoftmaxLoss()
        return loss_fn

    def _setup_writter(self, path):
        path_writter = path
        summary_writer = tf.summary.create_file_writer(path_writter)
        return summary_writer

    @tf.function
    def _training_step(self, iter_train):
        inputs, label = next(iter_train)

    @tf.function
    def _testing_step(self, iter_test):
        raise NotImplementedError

    def training(self):
        self.optimizer = self._setup_optimizer
        self.loss_fn = self._setup_loss
        self.writter_train = self._setup_writter(path="")
        self.writter_vali = self._setup_writter(path="")

        train_dataset = iter(self.loader.train)