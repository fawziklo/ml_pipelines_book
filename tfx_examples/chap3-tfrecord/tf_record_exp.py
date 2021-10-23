import tensorflow as tf

with tf.io.TFRecordWriter('../../tmp/test.tfrecord') as w:
    w.write(b"First Record")
    w.write(b"Second Record")

for record in tf.data.TFRecordDataset('test.chap3-tfrecord'):
    print(record)

