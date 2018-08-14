# Copyright 2018 Hooshmand Shokri, Daniel Hernandez. Columbia University.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
import numpy as np
import tensorflow as tf

from .datetools import addDateTime

from .generative.variational import ( ReparameterizedDistribution, MultiLayerPerceptron,
                                      LogitNormal )

def data_iterator_simple(Ydata, batch_size=1, shuffle=True):
    """
    """
    l_inds = np.arange(len(Ydata))
    if shuffle: 
        np.random.shuffle(l_inds)
    
    for i in range(0, len(Ydata), batch_size):
        yield Ydata[l_inds[i:i+batch_size]]


class VAE():
    """
    """
    def __init__(self, params):
        """
        """
        self.params = params
        
        with tf.variable_scope('VAE', reuse=tf.AUTO_REUSE):
            lat_dim = params.lat_dim
            batch_sz = params.batch_sz
            obs_dim = params.obs_dim
#             with_norm_flow = params.with_norm_flow
            num_enc_units = params.num_enc_units
            num_dec_units = params.num_dec_units
            shape_enc_units = [num_enc_units, num_enc_units//2, lat_dim]
            shape_dec_units = [num_dec_units, num_dec_units*2, obs_dim]
            num_mc_samps = params.num_mc_samps
            X = tf.placeholder(dtype=tf.float64, shape=[batch_sz, obs_dim], name='input')

            # Standard reparametrized normal
            encoder = ReparameterizedDistribution(tf.distributions.Normal, 
                                                  MultiLayerPerceptron,
                                                  input_tensor=X,
                                                  layers=shape_enc_units)
            z = encoder.sample(num_mc_samps)
            log_q = encoder.log_prob(z)

            # The decoder is a reparametrized logit_Normal because images [0., 1.]
            decoder = ReparameterizedDistribution(LogitNormal, MultiLayerPerceptron,
                                                  input_tensor=z, layers=shape_dec_units)
            x_hat = decoder.sample(1)
            # Prior distribution for codes.
            prior = tf.distributions.Normal(loc=np.zeros(lat_dim), scale=np.ones(lat_dim))

            # ELBO
            elbo = decoder.log_prob(X) + tf.reduce_mean(prior.log_prob(z), axis=-1) - log_q
#             self.elbo = tf.reduce_mean(elbo, axis=0)
            self.elbo = tf.reduce_mean(elbo)
            self.elbo_summ = tf.summary.scalar('ELBO', self.elbo)
            
            self.train_step = tf.get_variable("global_step", [], tf.int64,
                                              tf.zeros_initializer(), trainable=False)
            adagrad_trainer = tf.train.AdagradOptimizer(learning_rate=.01)
            self.train_op = adagrad_trainer.minimize(-self.elbo, global_step=self.train_step)
            #train_op = tf.train.AdamOptimizer(learning_rate=.01).minimize(- elbo)
            self.saver = tf.train.Saver(tf.global_variables())

    def train(self, datadict, rlt_dir=""):
        """
        """
        params = self.params
        
        Ytrain = datadict['Ytrain']
        Yvalid = datadict['Yvalid']
#         nvalid = Yvalid.shape[0]
        merged_summaries = tf.summary.merge([self.elbo_summ])
        self.writer = tf.summary.FileWriter(addDateTime('./logs/log'))
        
        valid_cost = -np.inf
        sess = tf.get_default_session()
        for ep in range(params.num_epochs):
            iterator_Y = data_iterator_simple(Ytrain, batch_size=params.batch_sz)
            elbo_avg = ctr = 0
            for batch in iterator_Y:
                _, elbo_value = sess.run([self.train_op, self.elbo],
                                            feed_dict={'VAE/input:0': batch})

                summaries = sess.run(merged_summaries, feed_dict={'VAE/input:0': batch})
                self.writer.add_summary(summaries, ep)                
                elbo_avg += elbo_value
                if not ctr % 500:
                    print('ELBO:', elbo_avg/500)
                    elbo_avg = 0

                    iter_Yvalid = data_iterator_simple(Yvalid[:500],
                                                       batch_size=params.batch_sz)
                    new_elbo_valid = sum([sess.run(self.elbo, feed_dict={'VAE/input:0': batch_v})
                                      for batch_v in iter_Yvalid])/500
                    if new_elbo_valid > valid_cost:
                        valid_cost = new_elbo_valid
                        print('ELBO(Valid):', valid_cost, '... Saving...')
                        self.saver.save(sess, rlt_dir+'vae', global_step=self.train_step)
                ctr += 1
  



#             if NORM_FLOW:
#                 encoder = FlowConditionalVariable(
#                     dim_x=lat_dim, y=x, flow_layers=N_LAYER, hidden_units=ENC_UNITS[:-1])
#                 z, log_q = encoder.sample_log_prob(n_samples=MC_EX)
#                 if batch_sz == 1:
#                     z = tf.expand_dims(z, axis=0)
#                     log_q = tf.expand_dims(log_q, axis=0)
#                 z = tf.transpose(z, [1, 0, 2])
#                 log_q = tf.transpose(log_q, [1, 0])
#             else:
