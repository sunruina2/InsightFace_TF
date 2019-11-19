import argparse
import os
import tensorflow as tf
from scipy import misc
import numpy as np
import io
import cv2


def to_rgb(img):
    if img.ndim < 3:
        h, w = img.shape
        ret = np.empty((h, w, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret
    else:
        return img


def augmentation(image, aug_img_size):
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.resize_images(image, [aug_img_size, aug_img_size])
    # image = tf.random_crop(image, ori_image_shape)
    return image


class ClassificationImageData:
    # https://www.jianshu.com/p/b480e5fcb638
    def __init__(self, img_size=112, augment_flag=True, augment_margin=16, write_path=''):
        self.img_size = img_size
        self.augment_flag = augment_flag
        self.augment_margin = augment_margin
        self.label_c_num = 0
        self.writer = tf.python_io.TFRecordWriter(write_path, options=None)
        self.label_n = 0

    def get_path_label(self, root):
        ids = list(os.listdir(root))
        if '.DS_Store' in ids:
            ids.remove('.DS_Store')
        if '.idea' in ids:
            ids.remove('.idea')
        ids.sort()  # 文件夹名字升序排序，对应peo_id_num序号0，1，2，...，label_n-1
        self.label_c_num = len(ids)  # 子级目录个数，即label人数
        id_dict = dict(zip(ids, list(range(self.label_c_num))))  # {每个人的文件夹名: peo_id_num}
        paths = []
        labels = []
        for i in ids:  # '00000'
            cur_dir = os.path.join(root, i)  # '/Users/finup/Desktop/rg/rg_game/data/training/Asian_s/00000'
            fns = os.listdir(
                cur_dir)  # <class 'list'>: ['00004.jpg', '00002.jpg', '00003.jpg', '00001.jpg', '00000.jpg']
            if '.DS_Store' in fns:
                fns.remove('.DS_Store')
            if '.idea' in fns:
                fns.remove('.idea')
            paths += [os.path.join(cur_dir, fn) for fn in fns]  # 每张照片对应的路径
            labels += [id_dict[i]] * len(fns)  # 每张照片对应的peo_id_num,与paths的下标一一对应,a = [1]*3 ,a = [1,1,1]
        return paths, labels  # 返回所有照片路径list“paths”；和每张照片对应的peo_id_num的list“labels”，len(paths)=len(labels)=293个样本，len(set(labels)) = 101人

    def image_processing(self, img):
        img.set_shape([None, None, 3])
        img = tf.image.resize_images(img, [self.img_size, self.img_size])

        if self.augment_flag:
            augment_size = self.img_size + self.augment_margin
            img = augmentation(img, augment_size)

        img = tf.cast(img, tf.float32) / 127.5 - 1

        return img

    def add_record(self, img, label, writer):
        print(label, type(label))
        img = to_rgb(img)
        img = cv2.resize(img, (self.img_size, self.img_size)).astype(np.uint8)
        shape = img.shape
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
            "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=list(shape))),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(tf_example.SerializeToString())  # 将Example中的map压缩为二进制文件

    def write_tfrecord_from_folders(self, read_dir):
        print('write tfrecord from folders...')
        paths, labels = self.get_path_label(read_dir)
        assert (len(paths) == len(labels))
        self.label_n = len(set(labels))
        total = len(paths)  # 样本数
        print('All sampls:', total)
        cnt = 0  # 写入计数
        for p, l in zip(paths, labels):
            b, g, r = cv2.split(cv2.resize(cv2.imread(p), (self.img_size, self.img_size)))
            imgcv_rgb = cv2.merge([r, g, b])
            self.add_record(imgcv_rgb, l, self.writer)
            cnt += 1
            if cnt % 1000 == 0:
                print('finish:', np.round(cnt / total, 2))
            # print('%d/%d' % (cnt, total), end='\r')
        print('done![%d/%d]' % (cnt, total))
        print('class num: %d' % self.label_c_num)

    def write_tfrecord_from_mxrec(self, read_dir):
        import mxnet as mx
        print('write tfrecord from mxrec...')
        idx_path = os.path.join(read_dir, 'ms1_train.idx')
        bin_path = os.path.join(read_dir, 'ms1_train.rec')
        imgrec = mx.recordio.MXIndexedRecordIO(idx_path, bin_path, 'r')
        s = imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        imgidx = list(range(1, int(header.label[0])))
        total = len(imgidx)
        cnt = 0
        labels = []
        for i in imgidx:
            img_info = imgrec.read_idx(i)
            header, img = mx.recordio.unpack(img_info)
            l = int(header.label)+(self.label_n)
            print('mxnet', l, type(l))
            labels.append(l)
            img = io.BytesIO(img)
            img = misc.imread(img).astype(np.uint8)
            self.add_record(img, l, self.writer)
            cnt += 1
            print('%d/%d' % (cnt, total), end='\r')
        self.label_c_num = len(set(labels))
        print('done![%d/%d]' % (cnt, total))
        print('class num: %d' % self.label_c_num)

    def write_close(self):
        self.writer.close()

    def parse_function(self, example_proto):
        dics = {
            'img': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
            'label': tf.FixedLenFeature(shape=(), dtype=tf.int64)
        }
        parsed_example = tf.parse_single_example(example_proto, dics)
        parsed_example['img'] = tf.decode_raw(parsed_example['img'], tf.uint8)
        parsed_example['img'] = tf.reshape(parsed_example['img'], parsed_example['shape'])
        return self.image_processing(parsed_example['img']), parsed_example['label']

    def read_TFRecord(self, filenames):
        dataset = tf.data.TFRecordDataset(filenames, buffer_size=256 << 20)
        return dataset.map(self.parse_function, num_parallel_calls=8)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='from which to generate TFRecord, folders or mxrec', default='mxrec')
    parser.add_argument('--image_size', type=int, help='image size', default=112)
    parser.add_argument('--read_dir', type=str, help='directory to read data', default='')
    parser.add_argument('--save_path', type=str, help='path to save TFRecord file', default='')

    return parser.parse_args()


if __name__ == "__main__":
    # # args = get_args()
    #
    # mode = 'folders'
    # image_size = 112
    # place_lst = ['Asian', 'African', 'Indian', 'Caucasian']
    #
    # # /Users/finup/Desktop/rg/train_data/Asian.tfrecord
    # # done![4947/4947]
    # # class num: 1728
    # #
    # # /Users/finup/Desktop/rg/train_data/African.tfrecord
    # # done![3863/3863]
    # # class num: 1543
    # #
    # # /Users/finup/Desktop/rg/train_data/Indian.tfrecord
    # # done![5182/5182]
    # # class num: 1923
    # #
    # # /Users/finup/Desktop/rg/train_data/Caucasian.tfrecord
    # # done![271354/271354]
    # # class num: 11326
    #
    # for i in place_lst:
    #
    #     read_dir = '/Users/finup/Desktop/rg/rg_game/data/training/' + i
    #     save_path = '/Users/finup/Desktop/rg/train_data/'+i+'.tfrecorda'
    #     print(save_path)
    #
    #     cid = ClassificationImageData(img_size=image_size)
    #     if mode == 'folders':
    #         cid.write_tfrecord_from_folders(read_dir, save_path)
    #     elif mode == 'mxrec':
    #         cid.write_tfrecord_from_mxrec(read_dir, save_path)
    #     else:
    #         raise ('ERROR: wrong mode (only folders and mxrec are supported)')
    import time
    st = time.time()
    # mode = 'folders'
    image_size = 112
    acele_dir = '/Users/finup/Desktop/rg/train_data/train_celebrity/celebrity_sample'
    ms1_dir = '/Users/finup/Desktop/rg/train_data/ms1_mxnet/'
    save_path = '/Users/finup/Desktop/rg/train_data/ms1_and_acele.tfrecords'
    '''样本数2830146'''
    # read_dir = '/data/sunruina/face_recognition/data_set/ms_celeb_arcpaper_tfrecords/train_data/train_celebrity/celebrity'
    # save_path = '/data/sunruina/face_recognition/data_set/ms_celeb_arcpaper_tfrecords/train_data/asian_cele.tfrecords'
    # '''office_avg'''
    # read_dir = '/data/sunruina/face_recognition/data_set/ms_celeb_arcpaper_tfrecords/train_data/dc_marking_trans_avg_k'
    # save_path = '/data/sunruina/face_recognition/data_set/ms_celeb_arcpaper_tfrecords/train_data/dc_marking_trans_avg_k.tfrecords'
    # print(save_path)

    cid = ClassificationImageData(img_size=image_size, write_path=save_path)

    cid.write_tfrecord_from_folders(acele_dir)
    cid.write_tfrecord_from_mxrec(ms1_dir)
    cid.write_close()
    print(np.round((time.time()-st)/60, 2))
