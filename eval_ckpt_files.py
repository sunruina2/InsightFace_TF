import tensorflow as tf
import argparse
from data.eval_data_reader import load_bin
from losses.face_losses import arcface_loss
from nets.L_Resnet_E_IR import get_resnet
import tensorlayer as tl
from verification import ver_test
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def get_args():
    parser = argparse.ArgumentParser(description='input information')
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='../ver_data', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')
    parser.add_argument('--num_output', default=85164, help='the image size')
    parser.add_argument('--batch_size', default=32, help='batch size to train network')
    parser.add_argument('--ckpt_file', default='../auroua_intf/InsightFace_iter_best_', type=str, help='the ckpt file path')
    parser.add_argument('--ckpt_index_list', default=['1950000.ckpt'], help='ckpt file indexes')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    ver_list = []
    ver_name_list = []
    for db in args.eval_datasets:
        print('begin db %s convert.' % db)
        data_set = load_bin(db, args.image_size, args)  # (data_list, issame_list)，len(data_list)=2 ,data_list[0].shape=(12000, 112, 112, 3), len(issame_list) = 6000
        ver_list.append(data_set)
        ver_name_list.append(db)

    images = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)  # (?, 112, 112, 3)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)  # (?, )
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)  # 随机初始化权重先把空架子搭起来，后续再往里面restore train好的权重
    net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)  # L_Resnet_E_IR (?, 112,112,3)>(?, 512)
    embedding_tensor = net.outputs
    # mv_mean = tl.layers.get_variables_with_name('resnet_v1_50/bn0/moving_mean', False, True)[0]
    # 3.2 get arcface loss
    logit = arcface_loss(embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=args.num_output)  # (?, 512)，(?,)，initializer ，85164

    gpu_config = tf.ConfigProto()  
    gpu_config.gpu_options.allow_growth = True 
    sess = tf.Session(config=gpu_config)
    saver = tf.train.Saver()

    result_index = []
    for file_index in args.ckpt_index_list:
        feed_dict_test = {}
        path = args.ckpt_file + file_index
        saver.restore(sess, path)
        print('ckpt file %s restored!' % file_index)
        feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
        feed_dict_test[dropout_rate] = 1.0
        results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=sess,
                           embedding_tensor=embedding_tensor, batch_size=args.batch_size, feed_dict=feed_dict_test,
                           input_placeholder=images)
        result_index.append(results)
    print(result_index)

