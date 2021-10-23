import csv

import tensorflow as tf
from tfx.components import ImportExampleGen

original_data_file = '../../tmp/heart.csv'
tfrecord_filename = 'heart2.tfrecords'
tf_record_writer = tf.io.TFRecordWriter(tfrecord_filename)


def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.encode()])
    )


with open(original_data_file) as csv_file:
    reader = csv.DictReader(csv_file, delimiter=",")
    for row in reader:
        example = tf.train.Example(features=tf.train.Features(feature={
            "age": _bytes_feature(row['age']),
            "sex": _bytes_feature(row['sex']),
            "trtbps": _bytes_feature(row['trtbps']),
        }))
        tf_record_writer.write(example.SerializeToString())
    tf_record_writer.close()

"""
exemple_gen= ImportExampleGen('heart.tfrecords')
print(exemple_gen)
"""


