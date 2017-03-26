import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

width = 28
height = 28
channels = 1
input_shape = width * height * channels

k = 1024
l = 2048
n_classes = 10

learning_rate = 1e-2

log_dir = 'log/'

mnist = input_data.read_data_sets('data', one_hot=True)
n_trains = 25
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

with tf.name_scope('ActualValue'):
    y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name='y_')

with tf.name_scope('InputLayer'):
    x = tf.placeholder(tf.float32, shape=[None, input_shape], name='x')

def layer(input, channels_in, channels_out, name='layer'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in, channels_out]),
                        name='W')
        b = tf.Variable(tf.truncated_normal([channels_out]),
                        name='B')

        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        return tf.matmul(input, w) + b

with tf.name_scope('NetworkModel'):
    y1 = tf.nn.relu(layer(x, input_shape, k))
    y2 = tf.nn.relu(layer(y1, k, l))
    y = layer(y2, l, n_classes)

with tf.name_scope('Train'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                            logits=y), name='loss')
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.name_scope('Accuracy'):
    correct_predictions = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32),
                name='Accuracy')

# Add scalar summaries
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)

init_op = tf.global_variables_initializer()
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # Visualization
    writer = tf.summary.FileWriter(log_dir + '1')
    writer.add_graph(sess.graph)

    sess.run(init_op)
    for n_train in range(n_trains):
        print("Training {}...".format(n_train))
        for n_batch in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _ = sess.run([train], feed_dict={x:batch_x, y_:batch_y})
            if n_batch % 5 == 0:
                s = sess.run(summary_op, feed_dict={x:mnist.test.images,
                             y_:mnist.test.labels})
                writer.add_summary(s, n_train*total_batch+n_batch)
