from embark.tensorflow.data_provider import DataProvider
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


class Mnist(DataProvider):
    def __init__(self, batch_size,
                 num_readers=10,
                 num_batchers=5,
                 num_fetchers=5,
                 **kwargs):
        self.input_shapes = [[28, 28], [10]]
        self.input_names = ['image', 'label']
        self.input_types = [tf.int32, tf.int32]
        super(Mnist, self).__init__(batch_size,
                                    num_readers=num_readers,
                                    num_batchers=num_batchers,
                                    num_fetchers=num_fetchers)

    def add_data_source(self, data_source, max_num_examples=None):
        use_train = data_source == 'train'
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        source = mnist.train if use_train else mnist.test
        num_datas = max_num_examples if max_num_examples is not None else len(source)
        for i in range(num_datas):
            image = source.images[i]
            image = np.reshape((image > .5).astype(int), self.input_shapes[0])
            label = source.labels[i]
            self.data.append([image, label])

    def _read_data(self, actual_data):
        """

        :param actual_data: 
        :return: 
        """
        return actual_data

    def _build_inputs(self):
        return [tf.placeholder(dtype, shape, name=name) for name, shape, dtype in
                zip(self.input_names, self.input_shapes, self.input_types)], self.input_names


data_provider = Mnist
