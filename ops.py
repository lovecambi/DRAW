# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:42:20 2017
@author: fankai
"""

import tensorflow as tf
import numpy as np
import math

def linear(x, output_dim, name='linear'):
    """
    x : batch_size * input_dim
    affine transformation Wx+b
    """
    input_dim = x.get_shape().as_list()[1]
    thres = np.sqrt(6.0 / (input_dim + output_dim))
    W = tf.get_variable("W", [input_dim, output_dim], initializer=tf.random_uniform_initializer(minval=-thres, maxval=thres)) 
    b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, W) + b


def conv2d(x, x_filters, n_filers,
           k_h=5, k_w=5, stride_h=2, stride_w=2, stddev=0.02, 
           bias=True, padding='SAME', name='conv2d'):
#    input_size = k_h * k_w * x_filters
#    W_initializer = tf.random_uniform_initializer(-1.0 / math.sqrt(input_size), 1.0 / math.sqrt(input_size))
#    W = tf.get_variable('W', [k_h, k_w, x_filters, n_filers], initializer=W_initializer)
    W = tf.get_variable('W', [k_h, k_w, x_filters, n_filers], initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(x, W, strides=[1, stride_h, stride_w, 1], padding=padding)
    if bias:
#        b = tf.get_variable('b', [n_filers], initializer=tf.constant_initializer(0.0))
        b = tf.get_variable('b', [n_filers], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = conv + b
    return conv


def deconv2d(x, output_shape,
             k_h=5, k_w=5, stride_h=2, stride_w=2, stddev=0.02,
             bias=True, padding='SAME', name='deconv2d'):
    in_C = x.get_shape().as_list()[-1]
#    input_size = k_h * k_w * in_C
#    W_initializer = tf.random_uniform_initializer(-1.0 / math.sqrt(input_size), 1.0 / math.sqrt(input_size))
#    W = tf.get_variable('W', [k_h, k_w, output_shape[-1], in_C], initializer=W_initializer)
    W = tf.get_variable('W', [k_h, k_w, output_shape[-1], in_C], initializer=tf.truncated_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride_h, stride_w, 1], padding=padding)
    if bias:
#        b = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        b = tf.get_variable('b', [output_shape[-1]], initializer=tf.truncated_normal_initializer(stddev=stddev))
        deconv = deconv + b
    return deconv
    

def lrelu(x, leaky=0.2):
    return tf.maximum(x, leaky*x)
