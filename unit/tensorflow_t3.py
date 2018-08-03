import tensorflow as tf
input1=tf.constant([2,3,5]);
input2=tf.constant([3,5,7]);
input3=tf.constant([9,6,3]);
r1=tf.add(input1,input2);
r2=tf.sub(input2,input1);
r3=tf.mul(input3,r2);
with tf.Session() as sess:
    result=sess.run([r1,r2,r3]);
    print(result);