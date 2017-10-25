import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/mnist_handwritten _digits",one_hot=True)

#
batch_size = 10
#
n_batch = mnist.train.num_examples // batch_size

#
x = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.float32,[None,10])

#
W = tf.Variable(tf.ones([28*28,10]))
b = tf.Variable(tf.ones([10]))

prediction = tf.nn.softmax(tf.matmul(x,W) + b)

#
loss =tf.reduce_mean(tf.square(y-prediction))

train_step =tf.train.AdagradOptimizer(.2).minimize(loss)
# AdagradOptimizer(.2).minimize(loss)
# GradientDescentOptimizer(0.5).minimize(loss)


#
init = tf.global_variables_initializer()
#
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# boolean -> float
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# train
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(200):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        if epoch % 10 == 0:
            acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
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
