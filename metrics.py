import tensorflow as tf

from tensorflow.contrib.metrics import streaming_true_negatives, streaming_false_positives, streaming_false_negatives, streaming_true_positives
def streaming_class_mean_iou(x, y, c):
    x_c = tf.equal(x, c)
    y_c = tf.equal(y, c)
    tp, tp_update = streaming_true_positives(x_c, y_c, name='true_positives')
    fn, tf_update = streaming_false_negatives(x_c, y_c, name='false_negatives')
    fp, fp_update = streaming_false_positives(x_c, y_c, name='false_positives')
    class_iou = tp / (fn + fp + tp)
    class_iou_update = [tp_update, tf_update, fp_update]
    return class_iou, class_iou_update
