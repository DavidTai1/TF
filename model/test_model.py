#!/usr/bin/python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/mnist_handwritten _digits",one_hot=True)


x = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([28*28,2000],stddev = 0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1) + b1)
L1drop =tf.nn.dropout(L1,keep_prob)
#prediction = tf.nn.softmax(tf.matmul(x,W) + b)

# 2nd layer
W2 = tf.Variable(tf.truncated_normal([2000,2000],stddev = 0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1drop,W2) + b2)
L2drop =tf.nn.dropout(L2,keep_prob)

# 3rd layer
W3 = tf.Variable(tf.truncated_normal([2000,1000],stddev = 0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2drop,W3) + b3)
L3drop =tf.nn.dropout(L3,keep_prob)

# out layer
W4 = tf.Variable(tf.truncated_normal([1000,10],stddev = 0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L3drop,W4) + b4)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# boolean -> float
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "test.ckpt")
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0}))