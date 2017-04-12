import tensorflow as tf

from tensorflow.contrib.metrics import streaming_true_negatives, streaming_false_positives, streaming_false_negatives, streaming_true_positives
def streaming_class_mean_iou(x, y, c):
    x_c = tf.equal(x, c)
    y_c = tf.equal(y, c)
    tp, tp_update = streaming_true_positives(x_c, y_c, name='true_positives')
    fn, fn_update = streaming_false_negatives(x_c, y_c, name='false_negatives')
    fp, fp_update = streaming_false_positives(x_c, y_c, name='false_positives')
    # fn = tf.Print(fn, [fn], 'fn')
    # fp = tf.Print(fp, [fp], 'fp')
    # tp = tf.Print(tp, [tp], 'tp')

    # tp_update = tf.Print(tp_update, [tp_update], 'tp_update')
    # fn_update = tf.Print(fn_update, [fn_update], 'fn_update')
    # fp_update = tf.Print(fp_update, [fp_update], 'fp_update')
    class_iou = tp / (fn + fp + tp)
    class_iou_update = tf.group(tp_update, fn_update, fp_update)
    return class_iou, class_iou_update
