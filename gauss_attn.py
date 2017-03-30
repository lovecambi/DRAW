# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:42:55 2017

@author: fankai
"""

import tensorflow as tf

def GaussianAttention(img, raw_theta, output_size=[2,2], attn_mode='read', name='GaussianAttentionLayer', **kwargs):
    """Attention Layer with Gaussian filters
    Parameters:
        img : expect shape [batch_size, H, W, C]
        raw_theta : expect shape [batch_size, 4] represents [gx_, gy_, log_sigma2, log_delta, log_gamma]
            after non_linear transformation, theta = [gx, gy, sigma^2, delta, gamma]
        output_size : the size of attention patch if read mode
        mode : 'read' or 'write'
    
    Return:
        attention patch : expect shape [batch_size, attn_H, attn_W, C]
        intensity gamma : expect shape [batch_size, 1]
    """
    eps = 1e-8
    intput_size = img.get_shape().as_list()
    if attn_mode=='read':
        H = intput_size[1]
        W = intput_size[2]
        attn_H = output_size[0]
        attn_W = output_size[1]
    else:
        H = output_size[0]
        W = output_size[1]
        attn_H = intput_size[1]
        attn_W = intput_size[2]
        
    
    def filterbank(gx, gy, sigma2, delta):
        """
        (gx, gy): grid centre position
        sigma2: variance for gaussian filter
        delta: stride (distance between two grids)
        """
        grid_i = tf.reshape(tf.cast(tf.range(attn_W), tf.float32), [1, -1]) # 1 * attn_W, [0,1,...,attn_W-1]
        grid_j = tf.reshape(tf.cast(tf.range(attn_H), tf.float32), [1, -1])
        mu_x = gx + (grid_i - attn_W / 2 - 0.5) * delta # eq 19 tf can broadcast with (1, N) * (BS, 1) = (BS, N)
        mu_y = gy + (grid_j - attn_H / 2 - 0.5) * delta # eq 20
        a = tf.reshape(tf.cast(tf.range(W), tf.float32), [1, 1, -1]) # 1 * 1 * W, [0,1,...,W-1]
        b = tf.reshape(tf.cast(tf.range(H), tf.float32), [1, 1, -1]) # 1 * 1 * H, [0,1,...,H-1]
        mu_x = tf.reshape(mu_x, [-1, attn_W, 1])
        mu_y = tf.reshape(mu_y, [-1, attn_H, 1])
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])
        Fx = tf.exp(-tf.square(a - mu_x) / (2*sigma2) ) # batch_size x attn_W x W
        Fy = tf.exp(-tf.square(b - mu_y) / (2*sigma2) ) # batch_size x attn_H x H
        # normalize, sum over A and B dims
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), eps)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), eps)
        return Fx, Fy
        
    
    def attn_window(params):
        """
        params : raw_theta after linear transformation
        return Fx, Fy, gamma
        """
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(1, 5, params) # tf.unstack(params, axis=1)
        gx = (W + 1) / 2 * (gx_ + 1)
        gy = (H + 1) / 2 * (gy_ + 1)
        sigma2 = tf.exp(log_sigma2)
        delta = max( (W - 1) / (attn_W - 1), (H - 1) / (attn_H - 1) ) * tf.exp(log_delta) 
        return filterbank(gx, gy, sigma2, delta) + (tf.exp(log_gamma), )
        
    
    with tf.variable_scope(name):
        """
        img: batch_size x H x W x C (batch_size x attn_H x attn_W x C)
        Fx: batch_size x attn_W x W
        Fy: batch_size x attn_H x H
        return: batch_size x attn_H x attn_W x C (batch_size x H x W x C)
        """
        Fx, Fy, gamma = attn_window(raw_theta)
        if attn_mode == 'read':
            glimpse = [tf.batch_matmul(Fy, tf.batch_matmul(imgc, Fx, adj_y=True)) for imgc in tf.unstack(img, axis=3)]
            gamma = tf.reshape(gamma, [-1, 1, 1, 1])
        elif attn_mode == 'write':
            glimpse = [tf.batch_matmul(Fy, tf.batch_matmul(imgc, Fx), adj_x=True) for imgc in tf.unstack(img, axis=3)]
            gamma = tf.reshape(1.0/gamma, [-1, 1, 1, 1])
        glimpse = tf.stack(glimpse, axis=3) # batch_size x N x N x C
        return glimpse * gamma, Fx, Fy, gamma

    
