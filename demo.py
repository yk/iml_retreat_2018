#!/usr/bin/env python3

from absl import app
from absl import flags
from fountain.mnist import MNIST
import tensorflow as tf
import numpy as np
import tqdm

flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('epochs', 5, '')
args = flags.FLAGS


def main(_):
    data, labels = MNIST(False).get_data()
    labels = labels.astype(np.int64)
    N, D = data.shape[0], np.prod(data.shape[1:])

    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.repeat(args.epochs).shuffle(buffer_size=1000).batch(args.batch_size)
    iterator = dataset.make_one_shot_iterator()

    x, y = iterator.get_next()

    global_step = tf.get_variable('step', dtype=tf.int32, shape=(), trainable=False)

    x_flat = tf.reshape(x, (-1, D))
    w = tf.get_variable('w', (D, 10), tf.float32, tf.initializers.random_normal())

    logits = tf.matmul(x, w)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

    optim = tf.train.AdamOptimizer(2e-4)
    grads_and_vars = optim.compute_gradients(loss)
    train_op = optim.apply_gradients(grads_and_vars, global_step=global_step)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for epoch in range(args.epochs):
            print(f'Epoch {epoch}')
            num_batches = N // args.batch_size
            for _ in tqdm.trange(num_batches):
                _, loss_val, step_val = sess.run((train_op, loss, global_step))
            print(f'Step: {step_val}\t\tLoss: {loss_val}')


if __name__ == '__main__':
    app.run(main)
