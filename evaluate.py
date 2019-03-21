# -*- coding:utf-8 -*-
"""
    @describe:evaluate scripts with accuracy function and F1-score function
    @author:SevenHsu
    @date:2019-03-08
"""
import tensorflow as tf


def accuracy(logits, labels):
    """
    compute accuracy
    :param logits: predict logits output.[ndarray([])(batch_size,number_class)]
    :param labels: ground truth.[ndarray([])(batch_size,1)]
    :return: accuracy
    """
    prediction = tf.argmax(tf.nn.softmax(logits), 1, name='accuracy_prediction')
    correct_prediction = tf.equal(tf.argmax(labels, 1), prediction)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    return acc


def F1_score(logits, lables):
    """
    compute F1 score
    :param logits: predict logits output.[ndarray([])(batch_size,number_class)]
    :param lables: ground truth.[ndarray([])(batch_size,1)]
    :return:
    """
    pass
