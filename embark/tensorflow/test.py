import shutil
import os
import tensorflow as tf
import numpy as np
from embark.tensorflow.data_provider import DataProviderStartStopHook

slim = tf.contrib.slim
from embark.tensorflow.deploy import evaluation_loop, create_test_op

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_shape = [28, 28]
y_shape = [10]
num_gpus = 1
num_epochs = 10
tower_batch_size = 128
batch_size = tower_batch_size * num_gpus
num_examples = (5000 / batch_size) * (batch_size)
steps_per_epoch = num_examples / batch_size
num_steps = num_epochs * steps_per_epoch

data_source = [[np.reshape((mnist.train.images[i] > .5).astype(int), [28, 28]), mnist.train.labels[i]] for i in
               range(num_examples)]

checkpoint_dir = './train'
log_dir = checkpoint_dir

data_provider_class = NumericalDataProvider
model_class = ImageEncoderModel

graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('test') as scope:
        data_provider = data_provider_class(tower_batch_size, [x_shape, y_shape], ['x', 'y'], num_readers=1)
        data_provider.add_data_source(data_source)
        dp_hook = DataProviderStartStopHook(data_provider)

        test_op, model, summaries = create_test_op(model_class,
                                                   data_provider.data(),
                                                   num_gpus=num_gpus)

        summaries.append(model.end_epoch_summary())
        summary_op = tf.summary.merge(summaries)

        eval_interval_secs = 10
        config = tf.ConfigProto(allow_soft_placement=True)
        evaluation_loop(
            '',
            checkpoint_dir,
            log_dir,
            num_evals=steps_per_epoch,
            eval_op=test_op,
            summary_op=summary_op,
            hooks=[dp_hook],
            eval_interval_secs=eval_interval_secs,
            max_number_of_evaluations=10,
            session_config=config)
