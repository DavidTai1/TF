import tensorflow as tf
import numpy as np
# tf.Variable()  变量
# tf.constant()
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# state = tf.Variable(0,name="")
# new_v = tf.add(state,1)
# update = tf.assign(state,new_value)
# with tf.Session() as sess:
#   sess.run(init)
#   print(sess.run(state))
#   for _ in range(5):
#       sess.run(update)
#       print(sess.run(state)) 1,2,3,4,5
#
# Fetch
# input1 = xxx
# input2 = xxx
# input3 = xxx
# add = tf.add(input1,input2)
# mul = tf.multiply(input3,add)
# sess = tf.Session()
# result = sess.run([mul,add])
# ---
# Feed
# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
# output = tf.multiply(input1,input2)
# with tf.Session() as sess:
#   print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))

x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2
print(y_data)

b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

loss = tf.reduce_mean(tf.square(y_data-y))

optimizer = tf.train.GradientDescentOptimizer(0.2)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step,sess.run([k,b]))
