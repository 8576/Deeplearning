# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnistdata', one_hot=True)

input_size = 784
unit_size_1 = 256
unit_size_2 = 128
n_class = 10
batch_size = 100
epoch_num = 200
input_x = tf.placeholder(dtype=tf.float32,
                         shape=[None, input_size],
                         name='input_x')
y = tf.placeholder(dtype=tf.float32,
                   shape=[None, n_class],
                   name='y')

def layer(input_data, input_size, output_size, active_fun=None):
    # input_data 输入
    # input_size 上一层呢神经元的数目
    # output_size 当前层神经元的数目


    w = tf.get_variable(name='w', dtype=tf.float32,
                        shape=[input_size, output_size],
                        initializer=tf.random_normal_initializer())
    b = tf.get_variable(name='b', dtype=tf.float32,
                        initializer=tf.random_normal_initializer(),
                        shape=[output_size])
    output = tf.add(tf.matmul(input_data, w), b)
    if active_fun == None:
        out_put = output
    else:
        out_put = active_fun(output)
    return out_put

def build_net():
    with tf.variable_scope('layer1'): # 隐藏层
        layer1 = layer(input_x, input_size, unit_size_1, tf.nn.sigmoid)

    with tf.variable_scope('layer2'): # 隐藏层
        layer2 = layer(layer1, unit_size_1, unit_size_2, tf.nn.sigmoid)

    with tf.variable_scope('layer3'): #输出层
        logits = layer(layer2,unit_size_2, n_class)
    # logits = tf.nn.sigmoid_cross_entropy_with_logits(outlayer)
    return logits

logits = build_net()

# 反向过程
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
# 测试
acc_1 = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
acc = tf.reduce_mean(tf.cast(acc_1, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epoch = 0
    while epoch < epoch_num:
        sum_loss = 0
        avg_loss = 0
        batch_num = mnist.train.num_examples // batch_size
        # batch_num 是一个epoch训练的次数
        for i in range(batch_num):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feeds = {input_x: batch_xs, y:batch_ys}
            sess.run(train, feed_dict=feeds)
            sum_loss += sess.run(loss, feeds)
        avg_loss = sum_loss /batch_num

