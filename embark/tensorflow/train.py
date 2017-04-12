import tensorflow as tf

slim = tf.contrib.slim
import numpy as np
import embark.tensorflow.deploy as deploy
from embark.tensorflow.deploy_configuration import DeployConfiguration

tf.logging.set_verbosity(tf.logging.INFO)
from tensorflow.examples.tutorials.mnist import input_data

config_file = './configs/mnist_autoencoder.json'
config = DeployConfiguration(config_file)

tower_batch_size = config.get_tower_batch_size()
num_gpus = config.get_num_gpus()


global_step = slim.get_or_create_global_step()

with tf.name_scope('train') as scope:
    data_source = config.get_data_source_config('train')
    data_provider = deploy.create_data_provider(config.get_data_provider_config(),
                                              tower_batch_size,
                                              data_source)

    num_examples = config.get_operation_config()['num_examples']
    num_epochs = config.get_operation_config()['num_epochs']
    examples_per_epoch = data_provider.size()

    num_steps, steps_per_epoch = deploy.get_steps(examples_per_epoch,
                                                  num_gpus,
                                                  num_examples,
                                                  num_epochs,
                                                  tower_batch_size)

    optimizer = deploy.create_optimizer(config.get_optimizer_config(),
                                        config.get_learning_rate_config())
    train_op, model, summaries = deploy.create_train_op(model_config,
                                                        data_provider.data(),
                                                        optimizer,
                                                        num_gpus=num_gpus,
                                                        histograms=False)

    queue_summary = data_provider.get_queue_summary()
    summaries += [queue_summary]

    model_summary_op = [model.batch_summary()] if model.batch_summary() is not None else []
    summaries += model_summary_op

    summary_op = tf.summary.merge(summaries)

checkpoint_dir = './train2'
# Actually runs training.

train_step = deploy.create_train_step(data_provider, model, steps_per_epoch)

# Create the initial assignment op
init_fn = deploy.create_restore_fn(config.get_restore_config())
# init_fn = None

import shutil

shutil.rmtree(checkpoint_dir, ignore_errors=True)
config = tf.ConfigProto(allow_soft_placement=True)
slim.learning.train(train_op,
                    checkpoint_dir,
                    train_step_fn=train_step,
                    number_of_steps=num_steps,
                    save_interval_secs=6,
                    save_summaries_secs=4,
                    summary_op=summary_op,
                    session_config=config,
                    init_fn=init_fn)
