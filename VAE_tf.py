#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zehao
"""

# Descriptions: Variational Autoencoder
# 
# Method: cost function = ELBO (evidence lower bound)
#		  encoder: generate u and ∑ --> N(u,∑) 				 			  
# 		  decoder: generate z, then as input to a NN
#		  VAE <--> posterior predictive sample

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
st = tf.contrib.bayesflow.stochastic_tensor
Normal = tf.contrib.distributions.Normal 
Bernoulli = tf.contrib.distributions.Bernoulli 


class DenseLayer(object):
  def __init__(self, M1, M2, activation=tf.nn.relu):
    self.M1 = M1
    self.M2 = M2
    self.W = tf.Variable(tf.random_normal(shape=(M1, M2)) * 2 / np.sqrt(M1))
    self.b = tf.Variable(np.zeros(M2).astype(np.float32))
    self.activation = activation

  def forward(self, X):
    return self.activation(tf.matmul(X, self.W) + self.b)

class VariationalAutoencoder(object):
  def __init__(self, D, hidden_layer_sizes):
    self.tfX = tf.placeholder(tf.float32, shape=(None, D))

    #  part one: encoder
    self.encoder_layers = []
    M_in = D
    for M_out in hidden_layer_sizes[:-1]:
      h = DenseLayer(M_in, M_out)
      self.encoder_layers.append(h)
      M_in = M_out
    ### final encoder size = M = input to the decoder size, hidden layer size
    M = hidden_layer_sizes[-1]
    h = DenseLayer(M_in, 2 * M, activation=lambda x: x)
    self.encoder_layers.append(h)

    ### generate tensor for mean and variance, variance use softplus function
    current_layer_value = self.tfX
    for layer in self.encoder_layers:
      current_layer_value = layer.forward(current_layer_value)
    self.means = current_layer_value[:, :M]
    self.stddev = tf.nn.softplus(current_layer_value[:, M:]) + 1e-6

    ### generate a stochastic tensor for Z 
    with st.value_type(st.SampleValue()):
      self.Z = st.StochasticTensor(Normal(loc=self.means, scale=self.stddev))
    ### Q(Z), the distribution of Z: self.Z.distribution

 

    # part two: decoder
    self.decoder_layers = []
    M_in = M
    for M_out in reversed(hidden_layer_sizes[:-1]):
      h = DenseLayer(M_in, M_out)
      self.decoder_layers.append(h)
      M_in = M_out

    ### Bernoulli accepts logits (pre-sigmoid) 
    ### so no activation function is needed at the final layer
    h = DenseLayer(M_in, D, activation=lambda x: x)
    self.decoder_layers.append(h)

    ### get the logits
    current_layer_value = self.Z
    for layer in self.decoder_layers:
      current_layer_value = layer.forward(current_layer_value)
    logits = current_layer_value
    posterior_predictive_logits = logits 

    ### get the output, and posterior predictive sample
    self.X_hat_distribution = Bernoulli(logits=logits) # distribution
    self.posterior_predictive_probs = tf.nn.sigmoid(logits) # probability sequence
    self.posterior_predictive = self.X_hat_distribution.sample() # sample


    #step3: prior predictive sample
    ### take sample from a Z ~ N(0, 1), and pass into the decoder
    ### same strcture as before
    standard_normal = Normal(
      loc=np.zeros(M, dtype=np.float32),
      scale=np.ones(M, dtype=np.float32)
    )

    Z_std = standard_normal.sample(1)
    current_layer_value = Z_std
    for layer in self.decoder_layers:
      current_layer_value = layer.forward(current_layer_value)
    logits = current_layer_value

    prior_predictive_dist = Bernoulli(logits=logits)
    self.prior_predictive = prior_predictive_dist.sample()
    self.prior_predictive_probs = tf.nn.sigmoid(logits)


    #step4:  prior predictive from input
    self.tfZ = tf.placeholder(tf.float32, shape=(None, M))
    current_layer_value = self.tfZ
    for layer in self.decoder_layers:
      current_layer_value = layer.forward(current_layer_value)
    logits = current_layer_value
    self.prior_predictive_from_input_probs = tf.nn.sigmoid(logits)


    # step5: cost function: ELBO
    kl = tf.reduce_sum(
      tf.contrib.distributions.kl_divergence(
        self.Z.distribution, standard_normal
      ),
      1
    )
    expected_log_likelihood = tf.reduce_sum(
      self.X_hat_distribution.log_prob(self.X),
      1
    )
    self.elbo = tf.reduce_sum(expected_log_likelihood - kl)
    
    # train_op
    self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-self.elbo)

    # set up session and variables for later, use IntersctiveSession
    self.init_op = tf.global_variables_initializer()
    self.sess = tf.InteractiveSession()
    self.sess.run(self.init_op)


  def fit(self, X, epochs=30, batch_sz=64):
    costs = []
    n_batches = len(X) // batch_sz
    for i in range(epochs):
      print("epoch:", i)
      np.random.shuffle(X)
      for j in range(n_batches):
        batch = X[j*batch_sz:(j+1)*batch_sz]
        _, c, = self.sess.run((self.train_op, self.elbo), feed_dict={self.X: batch})
        c /= batch_sz # just debugging
        costs.append(c)
        if j % 100 == 0:
          print("iteration: %d, cost: %.3f" % (j, c))
    plt.plot(costs)
    plt.show()

  def transform(self, X):
    return self.sess.run(
      self.means,
      feed_dict={self.X: X}
    )

  def prior_predictive_with_input(self, Z):
    return self.sess.run(
      self.prior_predictive_from_input_probs,
      feed_dict={self.Z_input: Z}
    )

  def posterior_predictive_sample(self, X):
    # returns a sample from p(x_new | X)
    return self.sess.run(self.posterior_predictive, feed_dict={self.X: X})

  def prior_predictive_sample_with_probs(self):
    # returns a sample from p(x_new | z), z ~ N(0, 1)
    return self.sess.run((self.prior_predictive, self.prior_predictive_probs))




