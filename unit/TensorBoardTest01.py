from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse

import tensorflow as tf

# 定义FLAGS用来传递全局参数
FLAGS = None


def main(_):
    # y = Wx + b， 初始化的时候随便定义一个初始值
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    # 输入值 x， 定义为占位符， 便于在学习过程中换成不同的值
    x = tf.placeholder(tf.float32)
    # 定义线性模型
    linear_model = tf.multiply(W, x) + b
    # 输出值 y， 定义为占位符， 便于在学习过程中换成不同的值
    y = tf.placeholder(tf.float32)

    # 损失loss，线性模型中以欧式距离来衡量损失值
    loss = tf.reduce_sum(tf.square(linear_model - y))
    # 定义优化器optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # 4个蓝色点的训练数据，分解成x和y的数组为
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]

    # 初始化Session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # 循环1000次，训练模型
        for i in range(1000):
            sess.run(train, {x: x_train, y: y_train})

        # 评估准确率
        curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
        print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

        # 保存计算图
        with tf.summary.FileWriter(FLAGS.summaryDir + 'train', sess.graph) as writer:
            writer.flush()


# 在运行main程序的时候，将参数传入执行代码中
# 本例中就指定了summaryDir参数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaryDir', type=str, default='',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main)