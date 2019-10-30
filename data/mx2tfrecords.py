import mxnet as mx
import argparse
import PIL.Image
import io
import numpy as np
import cv2
import tensorflow as tf
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='data path information'
    )
    parser.add_argument('--bin_path', default='../datasets/faces_ms1m_112x112/train.rec', type=str,
                        help='path to the binary image file')
    parser.add_argument('--idx_path', default='../datasets/faces_ms1m_112x112/train.idx', type=str,
                        help='path to the image index path')
    parser.add_argument('--tfrecords_file_path', default='../datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    args = parser.parse_args()
    return args


def mx2tfrecords_old(imgidx, imgrec, args):
    output_path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        encoded_jpg_io = io.BytesIO(img)
        image = PIL.Image.open(encoded_jpg_io)
        np_img = np.array(image)
        img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        img_raw = img.tobytes()
        label = int(header.label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    writer.close()


def mx2tfrecords(imgidx, imgrec, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        label = int(header.label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print(i, ' finished:', np.round(i / len(imgidx), 2))
    writer.close()


def parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([],
                                            tf.int64)}  # <class 'dict'>: {'image_raw': FixedLenFeature(shape=[], dtype=tf.string, default_value=None), 'label': FixedLenFeature(shape=[], dtype=tf.int64, default_value=None)}
    features = tf.parse_single_example(example_proto,
                                       features)  # <class 'dict'>: {'image_raw': <tf.Tensor 'ParseSingleExample/ParseSingleExample:0' shape=() dtype=string>, 'label': <tf.Tensor 'ParseSingleExample/ParseSingleExample:1' shape=() dtype=int64>}
    # You can do more image distortion here for training data
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))  # (112, 112, 3)
    r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)  # (112, 112, 1)，(112, 112, 1)，(112, 112, 1)
    img = tf.concat([b, g, r], axis=-1)  # (112, 112, 3)
    img = tf.cast(img, dtype=tf.float32)  # (112, 112, 3)
    img = tf.subtract(img, 127.5)  # (112, 112, 3)
    img = tf.multiply(img, 0.0078125)  # (112, 112, 3)
    img = tf.image.random_flip_left_right(img)  # (112, 112, 3)
    label = tf.cast(features['label'], tf.int64)
    return img, label


if __name__ == '__main__':

    st = time.time()
    # define parameters
    id2range = {}
    data_shape = (3, 112, 112)
    # args = parse_args()
    bin_path = '/Users/finup/Desktop/rg/train_data/ms1_mxnet/train.rec'  # path to the binary image file
    idx_path = '/Users/finup/Desktop/rg/train_data/ms1_mxnet/train.idx'  # path to the image index path
    tfrecords_file_path = '/Users/finup/Desktop/rg/train_data/ms1v2.tfrecords'  # path to the output of tfrecords file path

    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, bin_path, 'r')
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    # print(header.label)  # ???
    imgidx = list(range(1, int(header.label[0])))
    seq_identity = range(int(header.label[0]), int(header.label[1]))
    for identity in seq_identity:
        s = imgrec.read_idx(identity)
        header, _ = mx.recordio.unpack(s)
        a, b = int(header.label[0]), int(header.label[1])
        id2range[identity] = (a, b)

    print('label_c_n:', len(id2range))
    print('sample_n:', len(imgidx))
    # generate tfrecords
    mx2tfrecords(imgidx, imgrec, tfrecords_file_path)
    print('label_c_n:', len(id2range))
    print('sample_n:', len(imgidx))

    print('finish time:', int((time.time() - st) / 60))

    # config = tf.ConfigProto(allow_soft_placement=True)
    # sess = tf.Session(config=config)
    # # training datasets api config
    # dataset = tf.data.TFRecordDataset(tfrecords_file_path)
    # dataset = dataset.map(parse_function)
    # dataset = dataset.shuffle(buffer_size=30000)
    # dataset = dataset.batch(32)
    # iterator = dataset.make_initializable_iterator()
    # next_element = iterator.get_next()
    # # begin iteration
    # for i in range(1000):
    #     sess.run(iterator.initializer)
    #     while True:
    #         try:
    #             images, labels = sess.run(next_element)
    #             cv2.imshow('test', images[1, ...])
    #             cv2.waitKey(0)
    #         except tf.errors.OutOfRangeError:
    #             print("End of dataset")
