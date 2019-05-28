import numpy as np
import tensorflow as tf

"""
text, image = gen_captcha_text_and_image()
print  "验证码图像channel:", image.shape  # (60, 160, 3)
# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = len(text)
print   "验证码文本最长字符数", MAX_CAPTCHA  # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐
"""
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4
TRAIN_SIZE = 64
TEST_SIZE = 100


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""

# 文本转向量
char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
CHAR_SET_LEN = len(char_set)


def Read_data(filename="captcha/test.tfrecords", shape=[60, 160], batch_size=TRAIN_SIZE):
    """
        二进制读取从文件中读取图像数据
    :param filename: 文件名
    :param choose: 读取模式选择
    :return: image，label：返回二进制数据和图像标签
    """

    def parser(record):
        features = tf.parse_single_example(record, features={
            'label0': tf.FixedLenFeature([], tf.int64),
            'label1': tf.FixedLenFeature([], tf.int64),
            'label2': tf.FixedLenFeature([], tf.int64),
            'label3': tf.FixedLenFeature([], tf.int64),
            'img': tf.FixedLenFeature([], tf.string), })
        img = tf.decode_raw(features["img"], tf.uint8)  # 注意在这里只能是tf.uint8，tf.float32会报错
        img = tf.reshape(img, shape)
        # 归一化，转换到0-1之间
        # img = tf.cast(img, tf.float32) * (1. / 255.) - 0.5
        # img = tf.divide(tf.cast(img, tf.float32), 255.0)
        label0 = tf.cast(features["label0"], tf.int64)
        label1 = tf.cast(features["label1"], tf.int64)
        label2 = tf.cast(features["label2"], tf.int64)
        label3 = tf.cast(features["label3"], tf.int64)
        return img, [label0, label1, label2, label3,]

    # if choose==1:
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parser)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)  # 步长
    dataset = dataset.shuffle(
        buffer_size=1)  # batch(1)获取一张图像每次,buffer size=1，数据集不打乱；如果shuffle 的buffer size=数据集样本数量，随机打乱整个数据集
    iterator = dataset.make_one_shot_iterator()
    imglabelout = iterator.get_next()
    return imglabelout


def text2vec(labels, batch_size=TRAIN_SIZE):
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    for i in range(batch_size):
        vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        for j, label in enumerate(labels):
            idx = j * CHAR_SET_LEN + label[i]
            vector[idx] = 1
        batch_y[i, :] = vector
    return batch_y

#print text2vec('1aZ_')

# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


"""
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
"""


# 生成一个训练batch
# def get_next_batch(batch_size=128):
#     batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
#     batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
#
#     # 有时生成图像大小不是(60, 160, 3)
#     def wrap_gen_captcha_text_and_image():
#         while True:
#             text, image = gen_captcha_text_and_image()
#             if image.shape == (60, 160, 3):
#                 return text, image
#
#     for i in range(batch_size):
#         text, image = wrap_gen_captcha_text_and_image()
#         image = convert2gray(image)
#
#         batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
#         batch_y[i, :] = text2vec(text)
#
#     return batch_x, batch_y


####################################################################

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 32 * 40, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


# 训练
def train_crack_captcha_cnn():
    import time
    start_time = time.time()
    output = crack_captcha_cnn()
    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    lr = tf.Variable(0.0001, dtype=tf.float32)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    train_data = Read_data(filename='captcha/train.tfrecords', batch_size=TRAIN_SIZE)
    test_data = Read_data(filename='captcha/test.tfrecords', batch_size=TEST_SIZE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            img, labels = sess.run(train_data)
            labels = labels.reshape(4, -1)
            batch_y = text2vec(labels=labels, batch_size=TRAIN_SIZE)
            batch_x = (img.flatten() / 255).reshape(TRAIN_SIZE, -1)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), step, loss_)
            # 每100 step计算一次准确率
            if step % 20 == 0:
                # sess.run(tf.assign(lr, 0.001 * (0.95 ** step)))
                img_test, labels_test = sess.run(test_data)
                labels_test = labels_test.reshape(4, -1)
                batch_y_test = text2vec(labels=labels_test, batch_size=TEST_SIZE)
                batch_x_test = (img_test / 255).reshape(TEST_SIZE, -1)
                acc, loss_test = sess.run([accuracy, loss], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(u'***************************************************************第%s次的准确率为%s loss为' % (step, acc), loss_test)
                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.9:                  ##我这里设了0.9，设得越大训练要花的时间越长，如果设得过于接近1，很难达到。如果使用cpu，花的时间很长，cpu占用很高电脑发烫。
                    saver.save(sess, "crack_capcha.model", global_step=step)
                    print(time.time()-start_time)
                    break

            step += 1


train_crack_captcha_cnn()
