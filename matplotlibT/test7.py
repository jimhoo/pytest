'''
使用TensorFlow实现一个线性回归算法.
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 

 
print('~~~~~~~~~~开始设计计算图~~~~~~~~')
t_x=tf.constant([1,2,3,4,5,6,7,8,9,10])
t_y=tf.constant([2,4,6,5,9,13,14,18,21])



with tf.Graph().as_default():
    with tf.name_scope("Input"):
        X=tf.placeholder(tf.float32,name='x')
        Y=tf.placeholder(tf.float32,name="y")
    with tf.name_scope("Inference"):
        w=tf.Variable(np.random.randn(),name="weight");
        b=tf.Variable(np.random.randn(),name="bais");
        y_pred=tf.add(tf.multiply(X,w) , b);
    with tf.name_scope("loss"):
        TrainLoss = tf.reduce_mean(tf.pow((Y_pred - Y_true), 2))/2
    with tf.name_scope("Train"):
        Optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
        TrainOp=Optimizer.minimize(TrainLoss)
    with tf.name_scope("Evaluate"):
        EvalLoss=tf.reduce_mean(tf.pow((Y_pred - Y_true), 2)) / 2
        
print('启动会话，开启训练评估模式，让计算图跑起来')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("不断的迭代训练并测试模型")
    for step in range(10000):
        _, train_loss, train_w, train_b =sess.run([TrainOp,TrainLoss,w,b],feed_dict={X:t_x,Y:t_y});
        # 每隔几步训练完之后输出当前模型的损失
        if (step + 1) % 5 == 0:
            print("step:", '%04d' %(step+1), 'train_loss:','{:.9f}'.format(train_loss),'w=',tain_w,'b=',tain_b);
    print("训练结束!")
    w, b = sess.run([w, b])
    print("得到的模型参数：", "w=", w, "b=", b,)
    training_loss = sess.run(TrainLoss, feed_dict={X: train_X, Y_true: train_Y})
sess.close();