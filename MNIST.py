import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/mnist_handwritten _digits",one_hot=True)

#
batch_size = 100
#
n_batch = mnist.train.num_examples // batch_size

#
x = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
#

# W = tf.Variable(tf.ones([28*28,10]))
# b = tf.Variable(tf.ones([10]))
# better init
# 1st layer

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

#
# loss =tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

train_step =tf.train.AdagradOptimizer(.2).minimize(loss)
# AdagradOptimizer(.2).minimize(loss)
# GradientDescentOptimizer(0.5).minimize(loss)
#


#
init = tf.global_variables_initializer()
#
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# boolean -> float
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# train
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(50):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("#: "+str(epoch)+" accuracy: "+str(acc))

# AdagradOptimizer 0.2
#: 0 accuracy: 0.9156
#: 1 accuracy: 0.9213
#: 2 accuracy: 0.9243
#: 3 accuracy: 0.9264
#: 4 accuracy: 0.9263
#: 5 accuracy: 0.9276
#: 6 accuracy: 0.9268
#: 7 accuracy: 0.9301
#: 8 accuracy: 0.9285
#: 9 accuracy: 0.9294
#: 10 accuracy: 0.9298
#: 11 accuracy: 0.9295
#: 12 accuracy: 0.9291
#: 13 accuracy: 0.9289
#: 14 accuracy: 0.9307
#: 15 accuracy: 0.9307
#: 16 accuracy: 0.9298
#: 17 accuracy: 0.9289
#: 18 accuracy: 0.9295
#: 19 accuracy: 0.9308
#: 20 accuracy: 0.9304
#---------------------


# GradientDescentOptimizer 0.2
#: 0 accuracy: 0.9037
#: 1 accuracy: 0.9119
#: 2 accuracy: 0.9184
#: 3 accuracy: 0.9202
#: 4 accuracy: 0.9208
#: 5 accuracy: 0.9205
#: 6 accuracy: 0.9229
#: 7 accuracy: 0.9234
#: 8 accuracy: 0.926
#: 9 accuracy: 0.9269
#: 10 accuracy: 0.9256
#: 11 accuracy: 0.9266
#: 12 accuracy: 0.9268
#: 13 accuracy: 0.9271
#: 14 accuracy: 0.9268
#: 15 accuracy: 0.9275
#: 16 accuracy: 0.9283
#: 17 accuracy: 0.9278
#: 18 accuracy: 0.9289
#: 19 accuracy: 0.9289
#: 20 accuracy: 0.9289
#---------------------
