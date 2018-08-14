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
import os

# import matplotlib
# matplotlib.use('Agg')
# import seaborn as sns
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from src.vae import VAE

RUN_MODE = 'train'
DATASET = 'mnist'
MODEL = 'vae'
LOCAL_RLT_DIR = "/Users/danielhernandez/work_rslts/norm_flows/"
LOCAL_DATA_DIR = "./data/" 
THIS_DATA_DIR = 'allen/'
RESTORE_FROM_CKPT = False


LAT_DIM = 10
OBS_DIM = 100
ENC_UNITS = 256
DEC_UNITS = 128
MC_SAMPS = 5
NUM_EPOCHS = 5

BATCH_SZ = 1

flags = tf.app.flags
flags.DEFINE_string('mode', RUN_MODE, "The mode in which to run. Can be ['train', 'generate']")
flags.DEFINE_string('dataset', DATASET, "")
flags.DEFINE_string('model', MODEL, "")
flags.DEFINE_string('rlt_dir', LOCAL_RLT_DIR, "")
flags.DEFINE_boolean('restore_from_ckpt', RESTORE_FROM_CKPT, ("Should I restore a "
                                                "previously trained model?") )

flags.DEFINE_integer('lat_dim', LAT_DIM, "")
flags.DEFINE_integer('obs_dim', OBS_DIM, "")
flags.DEFINE_integer('num_enc_units', ENC_UNITS, "")
flags.DEFINE_integer('num_dec_units', DEC_UNITS, "")
flags.DEFINE_integer('num_mc_samps', MC_SAMPS, "")
flags.DEFINE_integer('num_epochs', NUM_EPOCHS, "")
flags.DEFINE_integer('batch_sz', BATCH_SZ, "")

params = tf.flags.FLAGS


def load_data(params):
    """
    """
    if params.model == 'vae':
        if params.dataset == 'mnist':
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
            mnist.train.images[mnist.train.images == 0.] += 0.001
            mnist.train.images[mnist.train.images == 1.] -= 0.001
            Ytrain = mnist.train.images
        else:
            pass
        
        params.obs_dim = Ytrain.shape[1]
        datadict = {'Ytrain' : Ytrain}
    
    return datadict

def build(params):
    """
    """
    model_class_dict = {'vae' : VAE}
    ModelClass = model_class_dict[params.model] 
    model = ModelClass(params)

    return model

def train(params):
    """
    """
    datadict = load_data(params)
    
    model = build(params)
    sess = tf.get_default_session()
    with sess:
        if params.restore_from_ckpt:
            saver = model.saver
            print("Restoring from ", params.load_ckpt_dir, " ...\n")
            ckpt_state = tf.train.get_checkpoint_state(params.load_ckpt_dir)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            print("Done.")
        else:
            sess.run(tf.global_variables_initializer())
        model.train(datadict)

def main(_):
    """
    """
    if params.mode == 'train':
        sess = tf.Session()
        with sess.as_default():
            train(params)


if __name__ == '__main__':
    tf.app.run()
    
    from sys import platform
    if platform == 'darwin':
        os.system('say "There is a beer in your fridge"')
