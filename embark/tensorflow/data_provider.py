import tensorflow as tf
from tensorflow.python.training import session_run_hook
from multiprocessing.pool import ThreadPool
from Queue import Queue

slim = tf.contrib.slim


class DataProvider(object):
    """
    Loads data in a queue.
    """

    def __init__(self, batch_size=12,
                 num_readers=10,
                 num_batchers=5,
                 num_fetchers=5,
                 **kwargs):
        self.batch_size = batch_size
        self.num_readers = num_readers
        self.num_batchers = num_batchers
        self.num_fetchers = num_fetchers

        self.reader_queue = None
        self.pool = None
        self.data = []
        self.inputs = None
        self.enqueue_data = None
        self.outputs = None
        self.queue_size_summary = None

        self._build()
        # queue that holds work to be read

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
        with tf.name_scope('data_provider') as scope:
            self.inputs, self.input_names = self._build_inputs()
            a = tf.constant([2])
            dtypes = [input.dtype for input in self.inputs]
            shapes = [input.shape.as_list() for input in self.inputs]

            input_queue_capacity = self.batch_size * 10
            self.input_queue = tf.FIFOQueue(capacity=input_queue_capacity,
                                            dtypes=dtypes,
                                            shapes=shapes,
                                            name='input_queue')
            self.input_queue_size = self.input_queue.size()
            self.input_queue_full = tf.cast(self.input_queue_size, tf.float32) * (1. / input_queue_capacity)
            with tf.name_scope('input_queue'):
                tf.summary.scalar("input_queue/fraction_of_%d_full" % input_queue_capacity, self.input_queue_full)

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
            self.queue_size_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))

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
                # print 'DP: session', session
                session.run(self.enqueue_data, feed_dict=feed_dict)
                # print 'DP: enqueued data'
                self.reader_queue.put(data)

    def get_queue_summary(self):
        return self.queue_size_summary

    def add_data_source(self, data_source, max_num_examples=None):
        """
        Parses an input datasource and adds individual data items to the internal
        data list.
        
        :param data_source: an undefined type that is understood by the subclass.
        :return: Nothing
        """
        raise NotImplementedError('Subclass must implement the method: add_data_source')

    def size(self):
        return len(self.data)

    def start(self, session):
        """
        Function to be run on a thread to enqueue data into the graph.
        
        :param data: single unit of data. 
        :return: 
        """

        # enqueue data
        self.pool = ThreadPool(processes=self.num_readers)
        self.reader_queue = Queue()
        for d in self.data:
            self.reader_queue.put(d)
        thread_sessions = [session for i in range(self.num_readers)]
        self.pool.map_async(self._enqueue, thread_sessions, chunksize=1)

    def stop(self, session):
        self.pool.terminate()
        # with session.as_default():
        #     session.run([self.input_queue.close(cancel_pending_enqueues=True),
        #                  self.prefetch_queue.close(cancel_pending_enqueues=True)])

    def data(self):
        return self.outputs


class DataProviderStartStopHook(session_run_hook.SessionRunHook):
    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.threads = []
        self.coord = None

    def after_create_session(self, session, coord):
        print 'Starting DP Hook: after_create_session'
        with session.as_default():
            self.data_provider.start(session)
        print 'Ending DP Hook: after_create_session'

    def end(self, session):
        print 'Starting DP Hook: end'
        self.data_provider.stop(session)
        print 'Ending DP Hook: end'
