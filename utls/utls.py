import tensorflow as tf


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
