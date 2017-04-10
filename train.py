import shutil
import os
import tensorflow as tf
from model import model
from data import load_data

slim = tf.contrib.slim


image_files = ['/nas/data/2017/02/23/23/22/00/cameras/cam3/images/1487892120035.png',
               '/nas/data/2017/02/23/23/22/00/cameras/cam3/images/1487892120085.png',
               '/nas/data/2017/02/23/23/22/00/cameras/cam3/images/1487892120135.png',
               '/nas/data/2017/02/23/23/22/00/cameras/cam3/images/1487892120185.png']


def main():
    checkpoint_dir = './output'
    log_dir = checkpoint_dir
    graph = tf.Graph()
    with graph.as_default():
        image_batch, y_batch = load_data(image_files, 2)

        predictions = model(image_batch)

        net_loss = slim.losses.mean_squared_error(predictions, y_batch)
        total_loss = slim.losses.get_total_loss(add_regularization_losses=True)

        tf.summary.scalar('losses/total_loss', total_loss)

        # Specify the optimization scheme:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=.000001)

        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        train_tensor = slim.learning.create_train_op(total_loss, optimizer)

        # Actually runs training.
        slim.learning.train(train_tensor, checkpoint_dir, number_of_steps = 20, save_interval_secs=6)

main()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())

