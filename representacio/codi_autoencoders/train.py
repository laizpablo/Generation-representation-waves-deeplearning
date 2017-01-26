#%reset
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functions import folder_exists
from datetime import datetime
import os
import tqdm # to see progress
tf.reset_default_graph() 

from audio_reader import AudioReader
from model import AutoEncoderModel

display_step = 50
save_model = 20000
logdir = 'logdir/AE_A/{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.now())
load_logdir = None
audio_dir = 'data'
sample_rate = 16000
batch_size = 8
histograms = True
realod_model = False
learning_rate = 1e-3
momentum = 0.9
_optimizer = 'adam'


# Input:
#   - learning_rate
#   - momentum
# Output:
#   - optimizer (can be one of the three next)
def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


def folder_exists (savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        
        
def load_model(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir), end="")

    # Returns CheckpointState proto from the "checkpoint" file.
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        # Restore variables from disk
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
    else:
        print(" No checkpoint found.")
        return None


def train_net(_net, reader, training_steps = None, error = 'squared'):
    
    logdir = 'logdir/AE_A/{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.now())
    
    with tf.device('/cpu:0'):
        
        if training_steps:
            training_epochs = training_steps
        else:
            training_epochs = int(1e4)
        
        folder_exists(logdir)
        
        # Create network. Parameters description in model.py
        net = _net
              
        if error == 'squared':
            reduced_loss = net.loss_squared_error()
        else:
            reduced_loss = net.loss_snr()

        optimizer = optimizer_factory[_optimizer]( learning_rate=learning_rate, momentum=momentum)

        trainable = tf.trainable_variables()

        optimizer = optimizer.minimize(reduced_loss, var_list=trainable)
        
        with tf.device('/cpu:0'): 
            writer = tf.train.SummaryWriter(logdir)
            writer.add_graph(tf.get_default_graph())
            run_metadata = tf.RunMetadata()
            summaries = tf.merge_all_summaries()

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        with tf.device('/cpu:0'):
            saver = tf.train.Saver(max_to_keep = 5, var_list = tf.trainable_variables())

        # Training cycle

        if realod_model:
            load_model(saver, sess, load_logdir)

        cost_snr = []
        number_errors = 0
        for epoch in tqdm.tqdm(xrange(training_epochs)):

            input_batch = reader.get_batch_train(batch_size)
            feed_dict = {net.x: input_batch, net.y: input_batch, net.keep_prob: 0.8}

            summary, loss, _ = sess.run([summaries, reduced_loss, optimizer], feed_dict)

            writer.add_summary(summary, epoch)

            if epoch % display_step == 1:
                cost_snr.append(loss) 


            if epoch % save_model == 0:
                #print "saving model: autoencoder_audio.ckpt-" + str(epoch) + "...",
                saver.save(sess, logdir + '/autoencoder_audio.ckpt', global_step=epoch)
                    

        print ("saving model:"+ logdir + "/autoencoder_audio.ckpt-" + str(epoch) + "...")
        saver.save(sess, logdir + '/autoencoder_audio.ckpt', global_step=epoch)
        print ("Optimization Finished!")
        print ("Total errors: " + str(number_errors))
    
    return sess, cost_snr

def load_sess(_net, reader, load_logdir, training_steps = None, error = 'squared'):

    with tf.device('/cpu:0'):
                
        # Create network. Parameters description in model.py
        net = _net
              
        if error == 'squared':
            reduced_loss = net.loss_squared_error()
        else:
            reduced_loss = net.loss_snr()

        optimizer = optimizer_factory[_optimizer]( learning_rate=learning_rate, momentum=momentum)

        trainable = tf.trainable_variables()

        optimizer = optimizer.minimize(reduced_loss, var_list=trainable)
        

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        with tf.device('/cpu:0'):
            saver = tf.train.Saver(max_to_keep = 5, var_list = tf.trainable_variables())

        # Training cycle

        load_model(saver, sess, load_logdir)
        return sess, saver
        
       