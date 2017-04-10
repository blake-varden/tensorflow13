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


# tag each input with a value
def main():
    checkpoint_dir = './output'
    log_dir = checkpoint_dir
    graph = tf.Graph()
    with graph.as_default():
        image_batch, y_batch = load_data(image_files, 2)

        predictions = model(image_batch)

        net_loss = slim.losses.mean_squared_error(predictions, y_batch)
        total_loss = slim.losses.get_total_loss(add_regularization_losses=True)

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'accuracy': slim.metrics.streaming_mean_absolute_error(predictions, y_batch)
        })

        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        summary_ops.append(tf.summary.scalar('eval_step', slim.get_or_create_global_step()))
        eval_interval_secs = 3

        slim.evaluation.evaluation_loop(
            '',
            checkpoint_dir,
            log_dir,
            num_evals=2,
            eval_op=names_to_updates.values(),
            summary_op=tf.summary.merge(summary_ops),
            eval_interval_secs=eval_interval_secs,
            max_number_of_evaluations=3)

main()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
