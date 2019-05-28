import tensorflow as tf
def Read_data(filename="train.tfrecords", shape=[224, 224, 3]):
    """
        二进制读取从文件中读取图像数据
    :param filename: 文件名
    :param choose: 读取模式选择
    :return: image，label：返回二进制数据和图像标签
    """

    def parser(record):
        features = tf.parse_single_example(record, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img': tf.FixedLenFeature([], tf.string), })
        img = tf.decode_raw(features["img"], tf.uint8)  # 注意在这里只能是tf.uint8，tf.float32会报错
        img = tf.reshape(img, shape)
        # 归一化，转换到0-1之间
        # img = tf.cast(img, tf.float32) * (1. / 255.) - 0.5
        img = tf.divide(tf.cast(img, tf.float32), 255.0)
        label = tf.cast(features["label"], tf.int64)
        return img, label

    # if choose==1:
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parser)
    dataset = dataset.repeat()
    dataset = dataset.batch(1)  # 步长
    dataset = dataset.shuffle(
        buffer_size=900)  # batch(1)获取一张图像每次,buffer size=1，数据集不打乱；如果shuffle 的buffer size=数据集样本数量，随机打乱整个数据集
    iterator = dataset.make_one_shot_iterator()
    imglabelout = iterator.get_next()
    return imglabelout


if __name__ == '__main__':
    datasetimg = Read_data()
    sess = tf.Session()
    for i in range(10):
        img, label = sess.run(datasetimg)
        print(img.shape, label)
    sess.close()
