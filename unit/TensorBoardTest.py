'''
Created on 2018年7月28日

@author: Administrator
'''
import tensorflow as tf

# y = Wx + b， 初始化的时候随便定义一个初始值
W=tf.Variable([.3], dtype=tf.float32)
b=tf.Variable([-.3],    dtype=tf.float32)
# 输入值 x， 定义为占位符， 便于在学习过程中换成不同的值
x=tf.placeholder(tf.float32)

linear_model=W*x+b

# 输出值 y， 定义为占位符， 便于在学习过程中换成不同的值
y=tf.placeholder(tf.float32)

# 损失loss，线性模型中以欧式距离来衡量损失值
loss=tf.reduce_sum(tf.square(linear_model-y))
# 定义优化器optimizer
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

# 4个蓝色点的训练数据，分解成x和y的数组为
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
# 循环1000次，训练模型
for i in range(1000):
    sess.run(train,{x:x_train, y:y_train})
    if i%10==0:
        cur_W,cur_b,cur_loss=sess.run([W,b,loss],{x:x_train,y:y_train})
        print("第i:%s次   W:%s b:%s loss:%s" %(i,cur_W,cur_b,cur_loss))
# 评估准确率
cur_W,cur_b,cur_loss=sess.run([W,b,loss],{x:x_train,y:y_train})
print("最佳 W:%s b:%s loss:%s" %(cur_W,cur_b,cur_loss))
with tf.summary.FileWriter(logdir='d:/',graph=tf.get_default_graph()) as writer:
    writer.flash()
writer.close()
sess.close()

