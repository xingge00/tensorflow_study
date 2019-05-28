import tensorflow as tf
import os
from PIL import Image



def get_image(data_path):
    filenames = [os.path.join(data_path, "data_batch_%d.bin" % i) for i in range(1, 6)]
    print(filenames)
    if check_cifar10_data_files(filenames) == False:
        exit()
    queue = tf.train.string_input_producer(filenames)
    return get_record(queue)


def get_record(queue):
    print('get_record')
    # 定义label大小，图片宽度、高度、深度，图片大小、样本大小
    label_bytes = 1
    image_width = 32
    image_height = 32
    image_depth = 3
    image_bytes = image_width * image_height * image_depth
    record_bytes = label_bytes + image_bytes

    # 根据样本大小读取数据
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(queue)

    # 将获取的数据转变成一维数组
    # 例如
    # source = 'abcde'
    # record_bytes = tf.decode_raw(source, tf.uint8)
    # 运行结果为[ 97  98  99 100 101]
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 获取label，label数据在每个样本的第一个字节
    label_data = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 获取图片数据，label后到样本末尾的数据即图片数据，
    # 再用tf.reshape函数将图片数据变成一个三维数组
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
        [3, 32, 32])

    # 矩阵转置，上面得到的矩阵形式是[depth, height, width]，即红、绿、蓝分别属于一个维度的，
    # 假设只有3个像素，上面的格式就是RRRGGGBBB
    # 但是我们图片数据一般是RGBRGBRGB，所以这里要进行一下转置
    # 注：上面注释都是我个人的理解，不知道对不对
    image_data = tf.transpose(depth_major, [1, 2, 0])

    return label_data, image_data


def check_cifar10_data_files(filenames):
    for file in filenames:
        if os.path.exists(file) == False:
            print('Not found cifar10 data.')
            return False
    return True


key, value = get_image('F:/Projects/PycharmProjects/dataset/cifar-10-batches-bin')

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # 这里才真的启动队列
    threads = tf.train.start_queue_runners(sess=sess)

    for i in range(20):
        # print("i:%d" % i)
        data = sess.run(value)
        lable = sess.run(key)
        print(i, lable)
        Image.fromarray(data).save('F:/Projects/PycharmProjects/test/' + '%d.jpg' % i)
