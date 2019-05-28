import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("F:/Projects/PycharmProjects/dataset/MNIST_data", one_hot=True)
img = Image.open('./images/4.png').convert('L').resize((28, 28), Image.BILINEAR)
array = np.asarray(img, dtype="float32").reshape(1, 784)

# 每个批次大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
# 定义学习率
lr = tf.Variable(0.001, dtype=tf.float32)


# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)#生成一个截断的正太分布
    return tf.Variable(initial)


# 初始化偏置值
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义样本
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


# 改变x的格式转为4D的向量
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一个卷基层的权值和偏置
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 把x_image和权值向量进行卷积，再加上偏置值，然后应用到relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)#进行max-pooling

# 初始化第二个卷积层和权值和偏置
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)#进行max-pooling


# 初始化第一个全连接层的权值
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# 把池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# 求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# keep_prob用来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 初始化第二个全连接层
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# 交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
# 其他优化器
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test = tf.argmax(prediction, 1)
# saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, 'net/my_net.ckpt')
    for epoch in range(1):
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1})
    print("训练结束，", "准确率:", acc)
    acc1 = sess.run(test, feed_dict={x: array,
                                     keep_prob: 1})
    print(acc1[0])

    # 保存模型
    # saver.save(sess, 'net/MNISTnet/my_net.ckpt')
