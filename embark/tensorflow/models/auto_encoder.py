import tensorflow as tf

slim = tf.contrib.slim
from embark.tensorflow.metrics import streaming_class_mean_iou
from embark.tensorflow.model import Model


class AutoEncoder(Model):
    def __init__(self, inputs, is_train=False, **kwargs):
        super(AutoEncoder, self).__init__(is_train=is_train)
        self.inputs = inputs
        self.metric_summary_ops = []
        self.reset_streaming_metric_variables = []
        self.names_to_updates = {}
        self.names_to_values = {}
        self._end_epoch_summary = None
        self._build()

    def _build(self):
        self._build_model()
        self._build_loss()
        self._build_metrics()
        self._build_update_ops()
        self._build_summary_ops()

    def _build_model(self):
        # model
        num_classes = 2

        self.x = tf.cast(self.inputs['x'], tf.int32)
        # self.x = tf.Print(self.x, [self.x], 'Input shape')
        x_shape = [-1] + [d.value for d in self.x.shape[1:]]
        flat = tf.cast(tf.reshape(self.x, [self.x.shape[0].value, -1]), tf.float32)
        encoded_500 = slim.fully_connected(flat, 500, scope='encoded_500')
        encoded_100 = slim.fully_connected(encoded_500, 100, scope='encoded_100')
        decoded_500 = slim.fully_connected(encoded_100, 500, scope='decoded_500')
        self.decoded = tf.reshape(slim.fully_connected(decoded_500, flat.shape[
            1].value * num_classes, scope='decoded'), x_shape + [num_classes], name='decoded_name')
        self.y = tf.cast(tf.arg_max(self.decoded, 3), tf.int32)

    def _build_loss(self):

        # loss
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.x, self.decoded)

    def _build_metrics(self):
        # metrics
        with tf.name_scope('metrics') as scope:
            self.names_to_values, self.names_to_updates = slim.metrics.aggregate_metric_map({
                "class_1_miou": streaming_class_mean_iou(self.x, self.y, 1),
            })
            scope = tf.get_variable_scope().name
            self.reset_streaming_metric_variables = [i for i in tf.local_variables() if str(i.name).startswith(scope)]

    def _build_summary_ops(self):
        # ops/summaries during batches

        self.metric_summary_ops = []
        for metric_name, metric_value in self.names_to_values.iteritems():
            op = tf.summary.scalar(metric_name, metric_value)
            # op = tf.Print(op, [metric_value], metric_name)
            self.metric_summary_ops.append(op)

        self.images = tf.cast(tf.expand_dims(self.y, -1) * 255, tf.uint8)
        self.image_summary = tf.summary.image('decoded', self.images, max_outputs=4)
        self.image_summary = tf.Print(self.image_summary, [tf.constant(True)],
                                      'Printing Image *****************************************')
        self.input_image_summary = tf.summary.image('input', tf.expand_dims(self.inputs['x'] * 255, -1), max_outputs=4)

        self._batch_summary = tf.summary.merge(self.metric_summary_ops) if self.is_train else None

        # ops/summaries end of epoch batches
        if self.is_train:
            self._end_epoch_summary = tf.summary.merge([self.image_summary, self.input_image_summary])
        else:
            self._end_epoch_summary = tf.summary.merge(
                self.metric_summary_ops + [self.image_summary, self.input_image_summary])

    def _build_update_ops(self):
        # begin epoch
        self._begin_epoch_ops = [tf.Print(tf.constant(True), [tf.constant(True)], 'local init'),
                                 tf.variables_initializer(self.reset_streaming_metric_variables)]
        self._batch_ops = self.names_to_updates.values()

    def begin_epoch_ops(self):
        # reset the streaming variables
        return self._begin_epoch_ops

    def batch_ops(self):
        # update the streaming variables
        return self._batch_ops

    def batch_summary(self):
        return self._batch_summary

    def end_epoch_summary(self):
        return self._end_epoch_summary

    def outputs(self):
        return {'y': self.y}

model = AutoEncoder

"""
Initialize locals every epoch

Run metric updates every step

Run summarioes every summary


"""
