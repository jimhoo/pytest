from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

x=tf.placeholder("float", [None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
#向量化后的图片x和权重矩阵W相乘 偏置b
#matmul乘积,softmax 线性回归模型
y=tf.nn.softmax(tf.matmul(x,W) +b )
#placeholder 占位符
y_=tf.placeholder("float",[None,10])
#reduce_sum为求和
cross_entropy=-tf.reduce_sum( y_*tf.log(y) )
# GradientDescentOptimizer 损失的梯度值 
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy);
init=tf.initialize_all_variables();
sess=tf.Session();
sess.run(init);
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100);
    result1=sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    
#tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配
correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(y_, 1))
#reduce_mean 求平均值
#reduce_max 求最大值
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"));
result=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
print(result)
