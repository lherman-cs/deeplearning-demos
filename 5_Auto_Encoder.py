import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
from os.path import join, splitext
from layers import layers
import config

train_dir = config.TRAIN_DIR
test_dir = config.TEST_DIR
n_trains = config.N_TRAINS
batch_size = config.BATCH_SIZE

width = config.WIDTH
height = config.HEIGHT
channels = config.CHANNELS
flat = config.FLAT
n_classes = config.N_CLASSES

k = 1000
l = 500
m = 30
n = l
o = k
num_imgs = 3

mnist = input_data.read_data_sets('data', one_hot=True)

def get_dict(train=True, batch=True):
    if train:
        if batch:
            batch_x, _ = mnist.train.next_batch(batch_size)
            return {x:batch_x}
        else:
            return {x:mnist.train.images}
    else:
        if batch:
            batch_x, _ = mnist.test.next_batch(batch_size)
            return {x:batch_x}
        else:
            return {x:mnist.test.images}

with tf.name_scope('InputLayer'):
    x = tf.placeholder(tf.float32, shape=[None, flat], name='x')

with tf.name_scope('NetworkModel'):
    with tf.name_scope('Encoder'):
        y1 = layers.ae_layer(x, flat, k)
        y2 = layers.ae_layer(y1, k, l)
        y3 = layers.ae_layer(y2, l, m)
    with tf.name_scope('Decoder'):
        y4 = layers.ae_layer(y3, m, n)
        y5 = layers.ae_layer(y4, n, o)
        y = layers.ae_layer(y5, o, flat)

with tf.name_scope('Train'):
    loss = tf.reduce_mean(tf.pow(y-x, 2), name='loss')
    train = tf.train.AdamOptimizer().minimize(loss)

with tf.name_scope('Accuracy'):
    accuracy = 1 - loss

# Add image summaries
x_img = tf.reshape(x, [-1, height, width, channels]) # input
y_img = tf.reshape(y, [-1, height, width, channels]) # reconstructed
tf.summary.image('InputImage', x_img, max_outputs=num_imgs)
tf.summary.image('OutputImage', y_img, max_outputs=num_imgs)

# Add scalar summaries
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)

init_op = tf.global_variables_initializer()
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # Open protocol for writing files
    train_writer = tf.summary.FileWriter(train_dir)
    train_writer.add_graph(sess.graph)
    test_writer = tf.summary.FileWriter(test_dir)

    sess.run(init_op)
    for n_train in range(1, n_trains+1):
        print("Training {}...".format(n_train))
        _ = sess.run([train], feed_dict=get_dict(train=True, batch=True))
        if n_train % 100 == 0:
            # Train
            s = sess.run(summary_op, feed_dict=get_dict(train=True, batch=False))
            train_writer.add_summary(s, n_train)
            # Test
            s = sess.run(summary_op, feed_dict=get_dict(train=False, batch=False))
            test_writer.add_summary(s, n_train)
