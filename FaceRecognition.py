import os

import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tqdm import tqdm

from loss.losses import SoftmaxLoss
from utls.utls import save_weight


class FaceRec(object):
    def __init__(self, model, loader, current_epochs, max_epochs, steps,
                 learning_rate, logs, saveweight_path):
        self.max_epochs = max_epochs
        self.current_epochs = current_epochs
        self.steps = steps
        self.model = model
        self.loader = loader

        self.lr = learning_rate
        self.logs = logs
        self.saveweight_path = saveweight_path

        # -------- setting up hyper parameter -------
        self.optimizer = None
        self.loss_fn = None
        self.writter_train = None
        self.writter_vali = None

    def _setup_optimizer(self):
        self.learning_rate = tf.constant(self.lr)
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)
        return optimizer

    def _setup_loss(self):
        loss_fn = SoftmaxLoss()
        return loss_fn

    def _setup_writter(self, path):
        path_writter = path
        summary_writer = tf.summary.create_file_writer(path_writter)
        return summary_writer

    def _setup_metrics_vali(self, names):
        self.metrics = {name: Mean() for name in names}

    def _training_step(self, iter_train):
        inputs, labels = next(iter_train)
        with tf.GradientTape() as tape:
            logist = self.model(inputs)
            reg_loss = tf.reduce_sum(self.model.losses)  # regularization_loss
            pred_loss = self.loss_fn(labels, logist)
            total_loss = pred_loss + reg_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return total_loss, pred_loss, reg_loss, self.optimizer.lr

    def _testing_step(self, iter_test):
        for index in tqdm(range(self.loader.steps_per_epoch), f'* Validate Epoch {self.current_epochs + 1}'):
            inputs, labels = next(iter_test)

            dict_values = {}
            logist = self.model(inputs)
            reg_loss = tf.reduce_sum(self.model.losses)  # regularization_loss
            pred_loss = self.loss_fn(labels, logist)
            total_loss = pred_loss + reg_loss

            dict_values['reg_loss'] = reg_loss
            dict_values['pred_loss'] = pred_loss
            dict_values['total_loss'] = total_loss

            metric_values = {name: float(value) for name, value in dict_values.items() if value is not None}
            for name, value in metric_values.items():
                self.metrics[name].update_state(value)

    def training(self):
        self.optimizer = self._setup_optimizer()
        self.loss_fn = self._setup_loss()
        self.writter_train = self._setup_writter(path=os.path.join(self.logs, 'train'))
        self.writter_vali = self._setup_writter(path=os.path.join(self.logs, 'validate'))

        train_dataset = iter(self.loader.train)
        validate_dataset = iter(self.loader.test)

        while self.current_epochs < self.max_epochs:
            self._setup_metrics_vali(names=['reg_loss', 'pred_loss', 'total_loss'])

            total_loss, pred_loss, reg_loss, lr_training = self._training_step(iter_train=train_dataset)

            if self.steps % 5 == 0:
                verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}"
                print(verb_str.format(self.current_epochs, self.max_epochs,
                                      self.steps % self.loader.steps_per_epoch,
                                      self.loader.steps_per_epoch,
                                      total_loss.numpy(),
                                      self.learning_rate.numpy()))

                # self._testing_step(iter_test=validate_dataset)

                # writter tensorboard
                with self.writter_train.as_default():
                    tf.summary.scalar(
                        'loss/total loss', total_loss, step=self.steps)
                    tf.summary.scalar(
                        'loss/pred loss', pred_loss, step=self.steps)
                    tf.summary.scalar(
                        'loss/reg loss', reg_loss, step=self.steps)
                    tf.summary.scalar(
                        'learning rate', lr_training, step=self.steps)

                # with self.writter_vali.as_default():
                #     tf.summary.scalar(
                #         'loss/total loss', self.metrics['total_loss'].result(), step=self.steps)
                #     tf.summary.scalar(
                #         'loss/pred loss', self.metrics['pred_loss'].result(), step=self.steps)
                #     tf.summary.scalar(
                #         'loss/reg loss', self.metrics['reg_loss'].result(), step=self.steps)

            if self.steps % 100 == 0:
                # saving weights
                name_save = 'e_{}_b_{}_{}.ckpt'.format(self.current_epochs,
                                                       self.steps % self.loader.steps_per_epoch,
                                                       total_loss.numpy())
                path_save = os.path.join(self.saveweight_path, name_save)
                save_weight(self.model, path_dir=path_save)

            self.steps += 1
            self.current_epochs = self.steps // self.loader.steps_per_epoch + 1

