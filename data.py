import shutil
import os
import tensorflow as tf
import threading

slim = tf.contrib.slim

#
# def load_image(filename_queue):
#     """
#     a
#     """
#     reader = tf.WholeFileReader()
#     image_file, image_serialized = reader.read(filename_queue)
#     image = tf.image.decode_png(image_serialized)
#     image.set_shape([384, 672, 3])
#     return image
#
#
# def load_data(image_files, batch_size):
#     """
#     """
#     filename_queue = tf.train.string_input_producer(image_files)
#     image = load_image(filename_queue)
#     y = tf.reshape(tf.reduce_sum(tf.cast(image, tf.float32)), [1])
#
#     min_after_dequeue = 2 * batch_size
#     capacity = min_after_dequeue + 3 * batch_size
#     image_batch, y_batch = tf.train.shuffle_batch(
#         [image, y], batch_size=batch_size, capacity=capacity,
#         min_after_dequeue=min_after_dequeue)
#
#     return image_batch, y_batch


# from multiprocessing import Queue, Pool, Value
from multiprocessing.pool import ThreadPool
from threading import Thread
from Queue import Queue


class DataProvider(object):
    """
    Loads data in a queue.
    """

    def __init__(self, batch_size,
                 num_readers=10,
                 num_batchers=5,
                 num_fetchers=5):
        self.batch_size = batch_size
        self.num_readers = num_readers
        self.pool = ThreadPool(processes=self.num_readers)
        self.num_batchers = num_batchers
        self.num_fetchers = num_fetchers
        self._build()
        # queue that holds work to be read
        self.reader_queue = Queue()

    def _build_inputs(self):
        """
        Args:
            None
    
        Returns:
            types: list of tensorflow types that will be dequeued
        """
        return [],

    def _build(self):
        """
        
        
        :param data: list of datas to be loaded. 
        :param batch_size: 
        :return: 
        """

        self.inputs, self.input_names = self._build_inputs()
        a = tf.constant([2])
        dtypes = [input.dtype for input in self.inputs]
        shapes = [input.shape.as_list() for input in self.inputs]

        input_queue_capacity = self.batch_size * 10
        self.input_queue = tf.FIFOQueue(capacity=input_queue_capacity,
                                        dtypes=dtypes,
                                        shapes=shapes,
                                        name='input_queue')
        self.enqueue_data = self.input_queue.enqueue(self.inputs, name='enqueue_inputs')
        inputs_dequeued = self.input_queue.dequeue(name='deqeue_inputs')

        batch_queue_capacity = self.batch_size * 10
        min_after_dequeue = self.batch_size * 2
        batch = tf.train.shuffle_batch(
            inputs_dequeued,
            self.batch_size,
            batch_queue_capacity,
            min_after_dequeue,
            num_threads=self.num_batchers,
            name='batch_queue')

        prefetch_capacity = 10
        self.prefetch_queue = slim.prefetch_queue.prefetch_queue(
            batch, capacity=prefetch_capacity)
        self.prefetch_queue_size = self.prefetch_queue.size()
        outputs = self.prefetch_queue.dequeue()
        self.outputs = dict(zip(self.input_names, outputs))

    def _read_data(self, data_pointer):
        """
        
        :param data_pointer: pointer to data that can be loaded.   
        :return: the loaded data
        """

    def _enqueue(self, session):
        with session.as_default():
            while True:
            	# print 'DP: getting data'
                data = self.reader_queue.get()
                # print 'DP: got data'
                read_data = self._read_data(data)
                # print 'DP: read data'
                feed_dict = dict(zip(self.inputs, read_data))
                session.run(self.enqueue_data, feed_dict=feed_dict)
                # print 'DP: enqueued data'
                self.reader_queue.put(data)

    def add_data_source(self, data_source):
        for d in data_source:
            self.reader_queue.put(d)

    def start(self, session):
        """
        Function to be run on a thread to enqueue data into the graph.
        
        :param data: single unit of data. 
        :return: 
        """

        # enqueue data
        thread_sessions = [session for i in range(self.num_readers)]
        self.pool.map_async(self._enqueue, thread_sessions, chunksize=1)

    def stop(self, session):
        self.pool.terminate()
        with session.as_default():
            session.run([self.input_queue.close(cancel_pending_enqueues=True),
                         self.prefetch_queue.close(cancel_pending_enqueues=True)])
        self.pool = ThreadPool(processes=self.num_readers)

    def data(self):
        return self.outputs


class NumericalDataProvider(DataProvider):
    def __init__(self, batch_size,
                 input_shapes,
                 input_names,
                 num_readers=10,
                 num_batchers=5,
                 num_fetchers=5):
        self.input_shapes = input_shapes
        self.input_names = input_names
        super(NumericalDataProvider, self).__init__(batch_size,
                                                    num_readers=num_readers,
                                                    num_batchers=num_batchers,
                                                    num_fetchers=num_fetchers)


    def _read_data(self, actual_data):
        """
        
        :param actual_data: 
        :return: 
        """
        return actual_data

    def _build_inputs(self):
        return [tf.placeholder(tf.float32, shape, name=name) for name, shape in zip(self.input_names, self.input_shapes)], self.input_names
