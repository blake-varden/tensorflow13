import shutil
import os
import tensorflow as tf

slim = tf.contrib.slim
from metrics import streaming_class_mean_iou


class Model(object):

    def __init__(self, is_train=False):
        self.is_train = is_train

    def _metric_prefix(self):
        return 'train/' if self.is_train else 'test/'

    def begin_epoch_ops(self):
        return []

    def begin_epoch_summary_ops(self):
        return []

    def batch_step_ops(self):
        return []

    def batch_step_summary_ops(self):
        return []

    def end_epoch_ops(self):
        return []

    def end_epoch_summary_ops(self):
        return []

    def outputs(self):
        pass


class ImageEncoderModel(Model):

    def __init__(self, inputs, is_train=False):
        super(ImageEncoderModel, self).__init__(is_train=is_train)
        self.inputs = inputs
        self.build()

    def build(self):

        # model
        num_classes = 2
        self.x = tf.cast(self.inputs['x'], tf.int32)
        x_shape = [-1] + [d.value for d in self.x.shape[1:]]
        flat = tf.cast(tf.reshape(self.x, [self.x.shape[0].value, -1]), tf.float32)
        encoded_500 = slim.fully_connected(flat, 500)
        encoded_100 = slim.fully_connected(encoded_500, 100)
        decoded_500 = slim.fully_connected(encoded_100, 500)
        decoded = tf.reshape(slim.fully_connected(decoded_500, flat.shape[
                             1].value * num_classes), x_shape + [num_classes])
        self.y = tf.cast(tf.arg_max(decoded, 3), tf.int32)

        # loss
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.x, decoded)

        # metrics
        with tf.variable_scope('metrics'):

            self.names_to_values, self.names_to_updates = slim.metrics.aggregate_metric_map({
                self._metric_prefix() + "class_1_miou": streaming_class_mean_iou(self.x, self.y, 1),
            })
            scope = tf.get_variable_scope().name
            self.reset_streaming_metric_variables = [i for i in tf.local_variables() if str(i.name).startswith(scope)]

        self.metric_summary_ops = []
        for metric_name, metric_value in self.names_to_values.iteritems():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            self.metric_summary_ops.append(op)

        # visualization
        self.images = tf.cast(tf.expand_dims(self.y, -1), tf.uint8)
        self.image_summary = tf.summary.image('decoded', self.images, max_outputs=4)

    def begin_epoch_ops(self):
        # reset the streaming variables

        return [tf.initialize_variables(v) for v in self.reset_streaming_metric_variables]

    def batch_step_ops(self):
        # update the streaming variables
        return self.names_to_updates.values()

    def batch_step_summary_ops(self):
        if self.is_train:
            return self.metric_summary_ops
        else:
            return []

    def end_epoch_summary_ops(self):
        if self.is_train:
            return [self.image_summary]
        else:
            return self.metric_summary_ops + [self.image_summary]

"""
Initialize locals every epoch

Run metric updates every step

Run summarioes every summary


"""
