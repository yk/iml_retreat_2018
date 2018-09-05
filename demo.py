#!/usr/bin/env python3

from absl import app
from absl import flags
from fountain.mnist import MNIST
import tensorflow as tf
import numpy as np
import tqdm

flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('epochs', 5, '')
flags.DEFINE_integer('log_every', 100, '')
args = flags.FLAGS


def main(_):
    data, labels = MNIST(False).get_data()
    N, D = data.shape[0], np.prod(data.shape[1:])

    x, y = tf.placeholder(tf.float32, (None, *data.shape[1:])), tf.placeholder(tf.int64, (None,))
    global_step = tf.get_variable('step', dtype=tf.int32, trainable=False)

    x_flat = tf.reshape(x, (-1, D))
    w = tf.get_variable('w', D, tf.float32, tf.initializers.random_normal())

    logits = tf.matmul(x, w)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    optim = tf.train.AdamOptimizer(2e-4)
    grads_and_vars = optim.compute_gradients(loss)
    train_op = optim.apply_gradients(grads_and_vars, global_step=global_step)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for epoch in range(args.epochs):
            print(f'Epoch {epoch}')
            data_idcs = np.arange(N)
            np.random.shuffle(data_idcs)
            num_batches = N // args.batch_size
            batch_idcs = data_idcs[:num_batches * args.batch_size].reshape((num_batches, args.batch_size))

            for batch_idx in tqdm.tqdm(batch_idcs):
                _, loss_val, step_val = sess.run((train_op, loss, global_step), feed_dict={
                    x: data[batch_idx],
                    y: data[batch_idx],
                })
                if step_val % args.log_every == 0:
                    print(f'Step: {step_val}\t\tLoss: {loss_val}')


if __name__ == '__main__':
    app.run(main)