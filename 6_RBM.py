import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

# Local
# from layers import layers
import config

model_name = config.MODEL_NAME
train_dir = config.TRAIN_DIR
test_dir = config.TEST_DIR
checkpoints_dir = config.CHECKPOINTS_DIR
n_trains = config.N_TRAINS
batch_size = config.BATCH_SIZE

width = config.WIDTH
height = config.HEIGHT
channels = config.CHANNELS
flat = config.FLAT
n_classes = config.N_CLASSES

k = 500
num_imgs = 3

mnist = input_data.read_data_sets('data', one_hot=True)


def get_dict(train=True, batch=True):
    if train:
        if batch:
            batch_x, _ = mnist.train.next_batch(batch_size)
            return {x: batch_x}
        else:
            return {x: mnist.train.images}
    else:
        if batch:
            batch_x, _ = mnist.test.next_batch(batch_size)
            return {x: batch_x}
        else:
            return {x: mnist.test.images}


with tf.name_scope('InputLayer'):
    x = tf.placeholder(tf.float32, shape=[None, flat], name='x')

with tf.name_scope('NetworkModel'):
    with tf.name_scope('rbm_layer'):
        # Encoder Variables
        w = tf.Variable(tf.truncated_normal([flat, k]),
                        name='W')
        _b = tf.Variable(tf.truncated_normal([k]), name='_B')

        # reconstructor Varables
        b = tf.Variable(tf.truncated_normal([flat]), name='B')

        _y = tf.nn.sigmoid(tf.matmul(x, w) + _b, name='_Y')
        y = tf.nn.sigmoid(tf.matmul(_y, tf.transpose(w)) + b, name='Y')

with tf.name_scope('Train'):
    loss = tf.reduce_mean(tf.pow(y - x, 2), name='loss')
    train = tf.train.AdamOptimizer().minimize(loss)

with tf.name_scope('Accuracy'):
    accuracy = 1 - loss

# Add image summaries
x_img = tf.reshape(x, [-1, height, width, channels])  # input
y_img = tf.reshape(y, [-1, height, width, channels])  # reconstructed
tf.summary.image('InputImage', x_img, max_outputs=num_imgs)
tf.summary.image('OutputImage', y_img, max_outputs=num_imgs)

# Add scalar summaries
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)

init_op = tf.global_variables_initializer()
summary_op = tf.summary.merge_all()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


def init():
    '''
    WARNING! This will override the trained model checkpoints.
    '''
    with tf.Session() as sess:
        # Open protocol for writing files
        train_writer = tf.summary.FileWriter(train_dir)
        train_writer.add_graph(sess.graph)
        test_writer = tf.summary.FileWriter(test_dir)

        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        sess.run(init_op)

        for n_train in range(1, n_trains + 1):
            print("Training {}...".format(n_train))
            _ = sess.run([train], feed_dict=get_dict(train=True, batch=True))
            if n_train % 100 == 0:
                saver.save(sess, os.path.join(checkpoints_dir, model_name),
                           global_step=n_train)

                # Train
                s = sess.run(summary_op, feed_dict=get_dict(train=True, batch=False))
                train_writer.add_summary(s, n_train)
                # Test
                s = sess.run(summary_op, feed_dict=get_dict(train=False, batch=False))
                test_writer.add_summary(s, n_train)


def load():
    with tf.Session() as sess:
        if os.path.exists(checkpoints_dir):
            saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))


init()
