import tensorflow as tf


def Read_data(filename="captcha/test.tfrecords", shape=[60, 160], batch_size=2):
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
        img = tf.divide(tf.cast(img, tf.float32), 255.0)
        label0 = tf.cast(features["label0"], tf.int64)
        label1 = tf.cast(features["label1"], tf.int64)
        label2 = tf.cast(features["label2"], tf.int64)
        label3 = tf.cast(features["label3"], tf.int64)
        return img, [label0, label1, label2, label3]

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


if __name__ == '__main__':
    datasetimg = Read_data()
    sess = tf.Session()
    for i in range(10):
        img, labels = sess.run(datasetimg)
        print(img.shape, labels)
    sess.close()
