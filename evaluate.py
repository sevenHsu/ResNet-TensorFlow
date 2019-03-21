# -*- coding:utf-8 -*-
"""
    @describe:evaluate scripts with accuracy function and F1-score function
    @author:SevenHsu
    @date:2019-03-08
"""
import tensorflow as tf


def accuracy(prediction, labels):
    """
    compute accuracy
    :param prediction: predict value.[ndarray([])(batch_size,1)]
    :param labels: ground truth.[ndarray([])(batch_size,num_classes)]
    :return: accuracy
    """
    correct_prediction = tf.equal(tf.argmax(labels, -1), prediction)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    return acc


def f1_score(prediction, labels):
    """
    compute F1 score
    :param prediction: predict logits output.[ndarray([])(batch_size,number_class)]
    :param labels: ground truth.[ndarray([])(batch_size,1)]
    :return:
    """
    with tf.name_scope("Macro_F1"):
        labels = tf.argmax(labels, 1)
        cm = tf.contrib.metrics.confusion_matrix(prediction, labels)
        acc = tf.diag_part(cm)
        true_positive = tf.reduce_sum(cm, axis=0)
        pred_positive = tf.reduce_sum(cm, axis=1)
        recall = acc / true_positive
        precision = acc / pred_positive
        f1 = 2 * recall * precision / (recall + precision)
        macro_f1 = tf.reduce_mean(f1)
    return macro_f1
