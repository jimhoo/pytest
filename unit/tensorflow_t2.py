import tensorflow as tf
x=tf.Variable([3,2]);
a=tf.constant([4,2]);
x1=tf.Variable(
    [
        [1, 2, 3],
        [3, 2, 1],
        [2, 3, 1]
        ]
    );
x2=tf.constant(
    [
        [11, 22, 33],
        [33, 22, 11],
        [22, 33, 11]
        ]
    );

sess=tf.InteractiveSession();
x.initializer.run();
x1.initializer.run();
sub=tf.sub(x,a);
sub1=tf.add(x1,x2);
print(sub.eval());
print(sub1.eval());
