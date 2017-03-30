# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 19:02:13 2017

@author: fankai
"""

import tensorflow as tf
import numpy as np
import os
import scipy.misc

from ops import *
from gauss_attn import GaussianAttention
#from glob import glob


class DRAW(object):
    """  

    """
    def __init__(self, img_shape, train_mode=True, model_path=None, 
                 read_attn=None, read_n=32, write_attn=None, write_n=8, attn_H=13, attn_W=13, attn_C=1,
                 T=8, lstm_hid_dim=256, latent_dim=100, batch_size=100, 
                 e_learning_rate=1e-3, eps=1e-8, 
                 grad_clip='Norm',
                 one_shot=True
                 ):
        
        # model parameters
        self.img_shape = img_shape
        self.train_mode = train_mode
        self.model_path = model_path
        
        self.read_attn = read_attn
        self.write_attn = write_attn
        self.H = img_shape[0]
        self.W = img_shape[1]
        self.C = img_shape[2]
        self.img_size = img_shape[0] * img_shape[1] * img_shape[2]

        self.gen_size = lstm_hid_dim
        self.enc_size = lstm_hid_dim
        self.z_size = latent_dim
        self.T = T
        self.batch_size = batch_size
        
        self.attn_H = attn_H
        self.attn_W = attn_W
        self.attn_C = attn_C
        
        self.read_n = read_n
        self.write_n = write_n
        self.write_size = self.write_n * self.write_n * self.attn_C
      
        self.e_learning_rate = e_learning_rate
        self.eps = eps
        
        # discriminator and generator RNN
        self.lstm_gen = tf.nn.rnn_cell.LSTMCell(self.gen_size, state_is_tuple=True)
        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(self.enc_size, state_is_tuple=True)

        # initial states and states variables
        self.c_prev_r = tf.zeros([self.batch_size, self.H, self.W, self.C])
        self.h_gen_prev_r = tf.zeros((self.batch_size, self.gen_size))
        self.gen_state_r = self.lstm_gen.zero_state(self.batch_size, tf.float32)
        self.enc_state_r = self.lstm_enc.zero_state(self.batch_size, tf.float32)
        self.h_gen_rs = [0] * self.T
        self.attn_rs = [0]*self.T
        
        self.enc_state_c = self.lstm_enc.zero_state(self.batch_size, tf.float32)
        self.x_cs = [0] * self.T
        
        self.c_prev_g = tf.zeros([self.batch_size, self.H, self.W, self.C])
        self.gen_state_g = self.lstm_gen.zero_state(self.batch_size, tf.float32)
        self.h_gen_gs = [0] * self.T
        self.x_gs = [0] * self.T
        
        # build model
        self.DO_SHARE = None
        self.x_r = tf.placeholder(tf.float32,shape=[self.batch_size] + list(self.img_shape)) 
        self.wu = tf.placeholder(tf.float32) # warm up
        self.Lz_r = 0.0
        
        for t in xrange(self.T):

            x_hat_r = self.x_r - tf.sigmoid(self.c_prev_r)
            rrt, self.attn_rs[t] = self.read(self.x_r, x_hat_r, self.h_gen_prev_r)
            
            if one_shot:            
                zr, mu, logsigma, sigma = self.sampleQ(tf.concat(1,[rrt, self.h_gen_prev_r]))
            else:
                h_enc_r, self.enc_state_r = self.encode(self.enc_state_r, tf.concat(1,[rrt, self.h_gen_prev_r]))
                zr, mu, logsigma, sigma = self.sampleQ(h_enc_r)
        
            self.h_gen_rs[t], self.gen_state_r = self.generate(self.gen_state_r, zr)
            wrt = self.write(self.h_gen_rs[t])
            self.c_prev_r += wrt
            
            self.x_cs[t] = tf.tanh(self.c_prev_r)
            self.h_gen_prev_r = self.h_gen_rs[t]

            self.Lz_r += 0.5*tf.reduce_mean(tf.reduce_sum(tf.square(mu) + tf.square(sigma) - 2*logsigma - 1, 1))
            
            self.DO_SHARE = True
            
            # generated image
            zg = tf.random_normal((self.batch_size, self.z_size), mean=0, stddev=1)
            h_gen_g, self.gen_state_g = self.generate(self.gen_state_g, zg)
            wgt = self.write(h_gen_g)
            self.c_prev_g += wgt
            self.x_gs[t] = tf.sigmoid(self.c_prev_g)

        self.Lx = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.c_prev_r, self.x_r), [1,2,3]))

        # compute total_e_loss: encode and decoder loss
        self.e_loss = self.Lx + self.wu * self.Lz_r
        
        t_vars = tf.trainable_variables()
        
        self.e_vars = [var for var in t_vars if 'g_' in var.name or 'e_' in var.name]
        
        self.e_optimizer = tf.train.AdamOptimizer(self.e_learning_rate, beta1=0.5, beta2=0.999)
        e_grads = self.e_optimizer.compute_gradients(self.e_loss, self.e_vars)
        clip_e_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in e_grads if grad is not None]
        self.e_optimizer = self.e_optimizer.apply_gradients(clip_e_grads)
    
    
    def train(self, train_set, valid_set, max_epoch=10, K=5):
        
        with tf.Session() as sess:
            
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())            

            for epoch in range(max_epoch):

                Lxs_, Lzs_, Les_ = [], [], []
                warm_up = 1.0 # min((0.0+epoch)/100.0, 1) 
                for train_batch in self.iterate_minibatches_u(train_set, self.batch_size, shuffle=True):
                    _, Le, Lx, Lz = sess.run([self.e_optimizer, self.e_loss, self.Lx, self.Lz_r], 
                                             feed_dict={self.x_r: train_batch, self.wu: warm_up})
                    Les_.append(Le)
                    Lxs_.append(Lx)
                    Lzs_.append(Lz)
                
                Lxs = np.mean(np.array(Lxs_), axis=0)
                Lzs = np.mean(np.array(Lzs_), axis=0)
                Les = np.mean(np.array(Les_), axis=0)
                        
                valid_Lbs = []
                for valid_batch in self.iterate_minibatches_u(valid_set, self.batch_size):
                    Lbv = sess.run(self.e_loss, feed_dict={self.x_r: valid_batch, self.wu: 1.0})
                    valid_Lbs.append(Lbv)
                valid_Lb = np.mean(np.array(valid_Lbs), axis=0) 
                
                print("Epoch=%d : Lx: %f Lz: %f Lb: %f Le: %f held-out Lb: %f" % (epoch,Lxs,Lzs,Lxs+Lzs,Les,valid_Lb))
                
                if epoch % 10 == 0 or epoch == max_epoch:
                    xshow = self.get_showimages(sess)
                    out_file = os.path.join(self.model_path,"draw_data"+str(epoch)+".npy")
                    np.save(out_file, xshow)
                    self.save_model(saver, sess, step=epoch)
                
            xshow = self.get_showimages(sess, self.batch_size)
            out_file = os.path.join(self.model_path,"draw_data_end.npy")
            np.save(out_file, xshow)


    def save_model(self, saver, sess, step):
        """
        save model with path error checking
        """
        if self.model_path is None:
            my_path = "model/" # default path in tensorflow saveV2 format
            # try to make directory
            if not os.path.exists("model"):
                try:
                    os.makedirs("model")
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
        else: 
            my_path = self.model_path + "/mymodel"
                
        saver.save(sess, my_path, global_step=step)
        
    
    def iterate_minibatches_u(self, datapath, batchsize, shuffle=False):
        """
        This function tries to iterate unlabeled data in mini-batch
        """
        if shuffle:
            indices = np.arange(len(datapath))
            np.random.RandomState(np.random.randint(1,2147462579)).shuffle(indices)
        for start_idx in xrange(0, len(datapath) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield datapath[excerpt]
        
    
    def get_showimages(self, sess,n=20):
        num_show = min(n, self.batch_size)
        xgs = sess.run(self.x_gs) # T*[batch_size x H x W x C]
        xshow_ = np.array(xgs)[:,:num_show,:,:,:] # T x num_show x H x W x C
        xshow = np.transpose(xshow_, [1,2,0,3,4]) # num_show x H x T x W x C
        xshow = 0.5*(xshow+1.0)
        return xshow.reshape([-1,self.T*self.W, self.C]) if self.C > 1 else xshow.reshape([-1,self.T*self.W])
        
    
    def encode(self, state, input):
        """
        run LSTM
        state = previous encoder state
        input = cat(read,h_dec_prev)
        returns: (output, new_state)
        """
        with tf.variable_scope("e_lstm",reuse=self.DO_SHARE):
            return self.lstm_enc(input,state)
            
    
    def generate(self, state, input):
        with tf.variable_scope("g_lstm",reuse=self.DO_SHARE):
            return self.lstm_gen(input, state)
        
        
    def write(self, h):
        with tf.variable_scope("g_write", reuse=self.DO_SHARE):
            w = linear(h, self.write_size)
            w = tf.reshape(w, [-1, self.write_n, self.write_n, self.attn_C])
#        with tf.variable_scope("g_write", reuse=self.DO_SHARE):
#            h_ = tf.reshape(h, [self.batch_size, 2, 2, int(self.gen_size/4)])
#            w = tf.nn.relu(deconv2d(h_, [self.batch_size, self.write_n, self.write_n, self.attn_C], 
#                            4, 4, 1, 1, padding='VALID')) # 5 x 5 
            
        if self.write_attn == None:
            with tf.variable_scope("g_deconv0", reuse=self.DO_SHARE):
                wr = tf.nn.relu(deconv2d(w, [self.batch_size, self.attn_H, self.attn_W, self.attn_C], 4, 4, 2, 2, padding='VALID'))
                                      
        elif self.write_attn == 'Gaussian':
            with tf.variable_scope("g_params", reuse=self.DO_SHARE):
                raw_params = linear(h, 5)
#                wr, _, _, _ = GaussianAttention(w, raw_params, [self.attn_H, self.attn_W], 'write')
                wr, _, _, _ = GaussianAttention(w, raw_params, [self.H, self.W], 'write')
                return wr
        
        else:
            raise NotImplementedError
        
        return self.write_deconv(wr)
    
    
    def write_deconv(self, x):
        with tf.variable_scope("g_deconv1", reuse=self.DO_SHARE):
            return deconv2d(x, [self.batch_size, 28, 28, self.C], 4, 4, 2, 2, padding='VALID')
        
        
    def read(self, x, x_hat, h_dec_prev):

        if self.read_attn == None:
            x_flat = tf.reshape(x, [self.batch_size, -1])
            x_hat_flat = tf.reshape(x_hat, [self.batch_size, -1])
            return tf.concat(0, [x_flat, x_hat_flat]), x_flat
            
        elif self.read_attn == 'Gaussian':
            with tf.variable_scope("e_writeG", reuse=self.DO_SHARE):
                raw_params = linear(h_dec_prev, 5)
                x_attn, Fx, Fy, gamma = GaussianAttention(x, raw_params, [self.read_n, self.read_n], 'read')
            x_hat = self.filter_img(x_hat, Fx, Fy) * gamma  # batch_size x read_n x read_n x C 
            x_flat = tf.reshape(x_attn, [self.batch_size, -1])
            x_hat_flat = tf.reshape(x_hat, [self.batch_size, -1])
            return tf.concat(1, [x_flat, x_hat_flat]), x_attn
            
        else:
            raise NotImplementedError
        

    ## Q-SAMPLER (VARIATIONAL AUTOENCODER) ##        
    def sampleQ(self, h):
        """
        Samples Zt ~ normrnd(mu,sigma^2) via reparameterization trick for normal dist
        mu is (batch,z_size)
        """
        et = tf.random_normal((self.batch_size, self.z_size), mean=0, stddev=1) # Vector noise
        with tf.variable_scope("e_mu", reuse=self.DO_SHARE):
            mu = linear(h, self.z_size)
        with tf.variable_scope("e_sigma", reuse=self.DO_SHARE):
            logsigma = linear(h, self.z_size)
            sigma = tf.exp(logsigma)
        return (mu + sigma*et, mu, logsigma, sigma) 
    
    
    def filter_img(self, img, Fx, Fy):
        """
        only for read:
            img : batch_size x H x W x C
            Fx : batch_size x N x W
            Fy : batch_size x N x H
            output : Fy*img*Fxt, batch_size x N x N x C
        """
        glimpse = [tf.batch_matmul(Fy, tf.batch_matmul(xc, Fx, adj_y=True)) for xc in tf.unstack(img, axis=3)]
        glimpse = tf.stack(glimpse, axis=3) # batch_size x N x N x C
        return glimpse
        
    
    def canvas_as_mavg(self, c, w, h):
        """
        moving average, c_t = u * c_{t-1} + (1 - u) * w_t
        """
        with tf.variable_scope("g_canvas", reuse=self.DO_SHARE):
             u = tf.sigmoid(linear(h, self.H * self.W)) # self.img_size
        u = tf.reshape(u, [-1, self.H, self.W, 1]) # 1 -> self.C
        return u * c + (1 - u) * w

    
    def print_vars_name(self, t_vars):
        for var in t_vars:
            print(var.name)
        

if __name__ == "__main__":
    # 28x28 @4x4@2x2 -> 13x13 @4x4@2x2 -> 5x5 
    # load data
    dataset = np.load('mnist_binarized.npz')
    train_data = np.reshape(dataset['X_train'], [-1, 28, 28, 1])
    valid_data = np.reshape(dataset['X_valid'], [-1, 28, 28, 1])
    test_data = np.reshape(dataset['X_test'], [-1, 28, 28, 1])
    
    mymodel = DRAW(img_shape=[28, 28, 1], train_mode=True, model_path="model_result",
                read_attn="Gaussian", read_n=5, write_attn="Gaussian", write_n=5, 
                T=64)
    mymodel.train(train_data,  test_data, max_epoch=500, K=1)
