import numpy as np
import tensorflow as tf
import time
from spinup.utils.logx import EpochLogger


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


# Simple script for training an MLP on MNIST.
def train_mnist(steps_per_epoch=100, epochs=5, 
                lr=1e-3, layers=2, hidden_size=64, 
                logger_kwargs=dict(), save_freq=1):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Load and preprocess MNIST data
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0

    # Define inputs & main outputs from computation graph
    x_ph = tf.placeholder(tf.float32, shape=(None, 28*28))
    y_ph = tf.placeholder(tf.int32, shape=(None,))
    logits = mlp(x_ph, hidden_sizes=[hidden_size]*layers + [10], activation=tf.nn.relu)
    predict = tf.argmax(logits, axis=1, output_type=tf.int32)

    # Define loss function, accuracy, and training op
    y = tf.one_hot(y_ph, 10)
    loss = tf.losses.softmax_cross_entropy(y, logits)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_ph, predict), tf.float32))
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Prepare session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, 
                                outputs={'logits': logits, 'predict': predict})

    start_time = time.time()

    # Run main training loop
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            idxs = np.random.randint(0, len(x_train), 32)
            feed_dict = {x_ph: x_train[idxs],
                         y_ph: y_train[idxs]}
            outs = sess.run([loss, acc, train_op], feed_dict=feed_dict)
            logger.store(Loss=outs[0], Acc=outs[1])

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state(state_dict=dict(), itr=None)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('Acc', with_min_and_max=True)
        logger.log_tabular('Loss', average_only=True)
        logger.log_tabular('TotalGradientSteps', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    train_mnist()