#placeholder 预留占位符
import tensorflow as tf

i1 = tf.placeholder(tf.float32)
i2 = tf.placeholder(tf.float32)
output = tf.mul(i1, i2)
with tf.Session() as sess:
    result=sess.run(output,feed_dict={i1:[2],i2:[3]});
    print(result);