import os
import tensorflow as tf
from PIL import Image
import numpy as np
import random

class_path = os.getcwd()+'/captcha/images/'

train_writer = tf.python_io.TFRecordWriter("captcha/train.tfrecords")
test_writer = tf.python_io.TFRecordWriter("captcha/test.tfrecords")

temp = os.listdir(class_path)
random.shuffle(temp)
for i, img_name in enumerate(temp):
    img_path = class_path + img_name
    img = Image.open(img_path)
    img = img.resize((60, 160))
    img = np.array(img.convert('L'))     #将图片灰度化
    img_raw = img.tobytes()              #将图片转化为原生bytes
    labels = img_name[0:4]
    num_labels = []
    for j in range(4):
        num_labels.append(int(labels[j]))
    example = tf.train.Example(features=tf.train.Features(feature={
        "label0": tf.train.Feature(int64_list=tf.train.Int64List(value=[num_labels[0]])),
        "label1": tf.train.Feature(int64_list=tf.train.Int64List(value=[num_labels[1]])),
        "label2": tf.train.Feature(int64_list=tf.train.Int64List(value=[num_labels[2]])),
        "label3": tf.train.Feature(int64_list=tf.train.Int64List(value=[num_labels[3]])),
        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    if i < 500:
        test_writer.write(example.SerializeToString())  # 序列化为字符串
    else:
        train_writer.write(example.SerializeToString())  # 序列化为字符串
train_writer.close()
test_writer.close()
