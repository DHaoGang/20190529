import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784],name='x-input')
    y = tf.placeholder(tf.float32,[None,10],name='y-input')
with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784,10]),name="w")
    with tf.name_scope('bias'):
        B = tf.Variable(tf.zeros([10]),name="b")
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W) + B
    with tf.name_scope('sotfmax'):
        prediction = tf.nn.softmax(wx_plus_b)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y - prediction))
with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
with tf.name_scope('accuracy'):
    with tf.name_scope('correction'):
        correction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    with tf.name_scope('accuracy1'):
        accuracy = tf.reduce_mean(tf.cast(correction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(1):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
        acc =sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter:"+str(epoch) + ",Testing Accuracy:" + str(acc))




