import os
import tensorflow as tf
from PIL import Image

cwd = os.getcwd()+'/train/'
for root, dirs, files in os.walk(cwd):
    print(dirs)  # 当前路径下所有子目录
    classes = dirs
    break

print(cwd)
writer = tf.python_io.TFRecordWriter("train.tfrecords")
for index, name in enumerate(classes):
    class_path = cwd + name + "/"
    print(class_path)
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = Image.open(img_path)
        img = img.resize((224, 224))
        if img.mode != 'RGB':
            print(img_path)
        img_raw = img.tobytes()              #将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  #序列化为字符串
writer.close()
