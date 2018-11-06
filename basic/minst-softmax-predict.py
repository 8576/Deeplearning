# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
dpath = r'D:\kaggledata\mnist\train.csv'

def loaddata(filename, ylabel=None):
    data = pd.read_csv(dpath, sep=',', header=0)
    if ylabel == 0:
        x, y = data.iloc[:, 1:-1], data.iloc[:, ylabel]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        return x_train, x_test, y_train, y_test
    else:
        x = data.iloc[:, :]
        return x

input_size = 784
n_class = 10
batch_size = 100
epoch_num = 30
unit_size_1 = 256
unit_size_2 = 256
unit_size_3 = 128
unit_size_4 = 128
unit_size_5 = 63

input_x = tf.placeholder(name='input_x',
                         shape=[None, input_size],
                         dtype=tf.float32)
y = tf.placeholder(name='y',
                   shape=[None, n_class],
                   dtype=tf.float32)


def linercomputer(input_data, input_size, output_size, active_fun=None):
    w = tf.get_variable(name='w', shape=[input_size, output_size],
                        initializer=tf.random_normal_initializer(),
                        dtype=tf.float32)
    b = tf.get_variable(name='b', shape=[output_size],
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer)
    outout = tf.add(tf.matmul(input_data, w), b)
    if active_fun:
        out_put = active_fun(outout)
    else:
        out_put = outout
    return out_put


def build_net():
    with tf.variable_scope('layer1'):
        layer1 = linercomputer(input_x, input_size,
                               unit_size_1,tf.nn.relu)
    with tf.variable_scope('layer2'):
        layer2 = linercomputer(layer1, unit_size_1,
                               unit_size_2, tf.nn.relu)
    with tf.variable_scope('layer3'):
        layer3 = linercomputer(layer2, unit_size_2,
                               unit_size_3, tf.nn.relu)
    with tf.variable_scope('layer4'):
        layer4 = linercomputer(layer3, unit_size_3,
                               unit_size_4, tf.nn.relu)
    with tf.variable_scope('layer5'):
        layer5 = linercomputer(layer4, unit_size_4,
                               unit_size_5, tf.nn.relu)
    with tf.variable_scope('layer6'):
        logist = linercomputer(layer5, unit_size_5, n_class)
    return logist


logits = build_net()
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
optimzer = tf.train.AdamOptimizer().minimize(loss)



def train():
    save = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x_train, x_test, y_train, y_test = loaddata(dpath, 0)
        print(x_train.shape)
        y_train = OneHotEncoder().fit_transform(y_train.values.reshape(-1, 1)).toarray()
        y_test = OneHotEncoder().fit_transform(y_test.values.reshape(-1, 1)).toarray()
        batch_num = int(x_train.shape[0] // batch_size)
        epoch = 0
        avg_loss = 0
        sum_loss = 0
        while epoch < epoch_num:
            for i in range(batch_num):
                feeds = {input_x: x_train.iloc[:, batch_size * i: batch_size * (i+1) - 1],
                         y: y_train[batch_size * i: batch_size * (i+1) - 1]}
                sess.run(optimzer, feed_dict=feeds)
                sum_loss += sess.run(loss, feed_dict=feeds)
                epoch += 1
            avg_loss = sum_loss / batch_num
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1),
                                              tf.argmax(y_test, 1)), tf.float32))
            accuary = sess.run(acc, feed_dict={input_x: x_test})
            print('epoch {} loss {} accuary{}'.format(epoch + 1, avg_loss, acc))


if __name__ == '__main__':
    train()

theta = theta - a *    (x.T.dot(x.dot(theta)) - y)
