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
    tf.enable_eager_execution()

    data, labels = MNIST(False).get_data()
    labels = labels.astype(np.int64)
    N, D = data.shape[0], np.prod(data.shape[1:])

    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.repeat(args.epochs).shuffle(buffer_size=1000).batch(args.batch_size)

    global_step = tf.get_variable('step', dtype=tf.int32, shape=(), trainable=False)
    w = tf.get_variable('w', (D, 10), tf.float32, tf.initializers.random_normal())
    optim = tf.train.AdamOptimizer(2e-4)

    for x, y in dataset:
        with tf.GradientTape() as tape:
            x_flat = tf.reshape(x, (-1, D))
            logits = tf.matmul(x, w)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

        w_grad = tape.gradient(loss, [w])
        grads_and_vars = zip(w_grad, [w])
        optim.apply_gradients(grads_and_vars, global_step=global_step)

        print(f'Step: {global_step.numpy()}\t\tLoss: {loss}')


if __name__ == '__main__':
    app.run(main)
