import math
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as K
import tensorflow as tf


def recall_at_k(y_true, y_pred, k=20):
    length = tf.gather(tf.shape(y_true), 1)
    k = tf.cond(length < k, lambda: length, lambda: k)
    y_pred, y_pred_ind_k = tf.nn.top_k(y_pred, k)
    y_true = tf.map_fn(lambda x: tf.gather(x[0], x[1]), (y_true, y_pred_ind_k), dtype=y_true.dtype)
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))
    return tp / (pp + K.epsilon())


def root_mean_square_error(y_true, y_pred):
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))


def precision(y_true, y_pred):
    """精确率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
    return tp / (pp + K.epsilon())


def recall(y_true, y_pred):
    """召回率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))  # possible positives
    return tp / (pp + K.epsilon())


def ndcg_at_k(y_true, y_pred, k=20):
    idcg_k = 0
    dcg_k = 0
    # n_k = k if K.shape(y_pred)[0] > k else K.shape(y_pred)[0]
    for i in range(k):
        idcg_k += 1 / K.log(i + 2)
    b1 = y_true
    b2 = y_pred
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    for c in range(K.shape(hits)[0]):
        dcg_k += 1 / K.log(hits[c][0] + 2, 2)
    return float(dcg_k / idcg_k)
