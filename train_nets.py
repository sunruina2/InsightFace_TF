import tensorflow as tf
import tensorlayer as tl
import argparse
from data.mx2tfrecords import parse_function
import os
# from nets.L_Resnet_E_IR import get_resnet
# from nets.L_Resnet_E_IR_GBN import get_resnet
from nets.L_Resnet_E_IR_fix_issue9 import get_resnet
from losses.face_losses import arcface_loss
from tensorflow.core.protobuf import config_pb2
import time
from data.eval_data_reader import load_bin
from verification import ver_test


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--net_depth', default=100, help='resnet depth, default is 50')
    parser.add_argument('--epoch', default=100000, help='epoch to train the network')
    parser.add_argument('--batch_size', default=32, help='batch size to train network')
    parser.add_argument('--lr_steps', default=[40000, 60000, 80000], help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    parser.add_argument('--weight_deacy', default=5e-4, help='learning alg momentum')
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--num_output', default=85164, help='the image size')
    parser.add_argument('--tfrecords_file_path', default='./datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=300, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=10000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=20, help='intervals to save ckpt file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # 1. define global parameters

    net_depth = 50  # resnet depth, default is 50
    epoch = 100000  # epoch to train the network
    batch_size = 64  # batch size to train network
    lr_steps = [40000, 60000, 80000]  # learning rate to train network
    momentum = 0.9  # learning alg momentum
    weight_deacy = 5e-4  # learning alg momentum
    eval_datasets = ['lfw']  # evluation datasets
    eval_db_path = '../ver_data'  # evluate datasets base path
    image_size = [112, 112]  # the image size
    num_output = 85164  # the image size
    tfrecords_file_path = '../train_data'  # path to the output of tfrecords file path
    summary_path = '../auroua_output/summary'  # the summary file save path
    ckpt_path = '../auroua_output/ckpt'  # the ckpt file save path
    log_file_path = '../auroua_output/logs'  # the ckpt file save path
    saver_maxkeep = 100  # tf.train.Saver max keep ckpt files
    buffer_size = 10000  # tf dataset api buffer size ?
    log_device_mapping = False  # show device placement log
    summary_interval = 300  # interval to save summary
    ckpt_interval = 10000  # intervals to save ckpt file
    validate_interval = 2000  # intervals to save ckpt file
    show_info_interval = 20  # intervals to save ckpt file
    
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    images = tf.placeholder(name='img_inputs', shape=[None, *image_size, 3], dtype=tf.float32)  # Tensor("img_inputs:0", shape=(?, 112, 112, 3), dtype=float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)  # Tensor("img_labels:0", shape=(?,), dtype=int64)
    # trainable = tf.placeholder(name='trainable_bn', dtype=tf.bool)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)
    # 2 prepare train datasets and test datasets by using tensorflow dataset api
    # 2.1 train datasets
    # the image is substracted 127.5 and multiplied 1/128.
    # random flip left right
    tfrecords_f = os.path.join(tfrecords_file_path, 'ms1.tfrecords')
    dataset = tf.data.TFRecordDataset(tfrecords_f)  # <TFRecordDatasetV1 shapes: (), types: tf.string>
    dataset = dataset.map(parse_function)  # map，parse_function函数对每一个图进行处理，bgr位置转换，标准化，随机数据增强
    dataset = dataset.shuffle(buffer_size=buffer_size)  # shuffle
    dataset = dataset.batch(batch_size)  # ((?, 112, 112, 3)， (?,))
    iterator = dataset.make_initializable_iterator()  # ((?, 112, 112, 3)， (?,))
    next_element = iterator.get_next()  # <class 'tuple'>: (<tf.Tensor 'IteratorGetNext:0' shape=(?, 112, 112, 3) dtype=float32>, <tf.Tensor 'IteratorGetNext:1' shape=(?,) dtype=int64>)
    # 2.2 prepare validate datasets
    ver_list = []
    ver_name_list = []
    for db in eval_datasets:
        print('begin db %s convert.' % db)
        data_set = load_bin(db, image_size, eval_db_path)  # [正(12000, 112, 112, 3), 反(12000, 112, 112, 3)], 6000
        ver_list.append(data_set)
        ver_name_list.append(db)
    # 3. define network, loss, optimize method, learning rate schedule, summary writer, saver
    # 3.1 inference phase
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = get_resnet(images, net_depth, type='ir', w_init=w_init_method, trainable=True, keep_rate=dropout_rate)
    # 3.2 get arcface loss
    logit = arcface_loss(embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=num_output)
    # test net  because of batch normal layer
    tl.layers.set_name_reuse(True)
    test_net = get_resnet(images, net_depth, type='ir', w_init=w_init_method, trainable=False, reuse=True, keep_rate=dropout_rate)
    embedding_tensor = test_net.outputs
    # 3.3 define the cross entropy
    inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
    # inference_loss_avg = tf.reduce_mean(inference_loss)
    # 3.4 define weight deacy losses
    # for var in tf.trainable_variables():
    #     print(var.name)
    # print('##########'*30)
    wd_loss = 0
    for weights in tl.layers.get_variables_with_name('W_conv2d', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(weights)
    for W in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/W', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(W)
    for weights in tl.layers.get_variables_with_name('embedding_weights', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(weights)
    for gamma in tl.layers.get_variables_with_name('gamma', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(gamma)
    # for beta in tl.layers.get_variables_with_name('beta', True, True):
    #     wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(beta)
    for alphas in tl.layers.get_variables_with_name('alphas', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(alphas)
    # for bias in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/b', True, True):
    #     wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(bias)

    # 3.5 total losses
    total_loss = inference_loss + wd_loss
    # 3.6 define the learning rate schedule
    p = int(512.0/batch_size)
    lr_steps = [p*val for val in lr_steps]
    print(lr_steps)
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001], name='lr_schedule')
    # 3.7 define the optimize method
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum)
    # 3.8 get train op
    grads = opt.compute_gradients(total_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)
    # train_op = opt.minimize(total_loss, global_step=global_step)
    # 3.9 define the inference accuracy used during validate or test
    pred = tf.nn.softmax(logit)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), dtype=tf.float32))
    # 3.10 define sess
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_mapping)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    # 3.11 summary writer
    summary = tf.summary.FileWriter(summary_path, sess.graph)
    summaries = []
    # # 3.11.1 add grad histogram op
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # 3.11.2 add trainabel variable gradients
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))
    # 3.11.3 add loss summary
    summaries.append(tf.summary.scalar('inference_loss', inference_loss))
    summaries.append(tf.summary.scalar('wd_loss', wd_loss))
    summaries.append(tf.summary.scalar('total_loss', total_loss))
    # 3.11.4 add learning rate
    summaries.append(tf.summary.scalar('leraning_rate', lr))
    summary_op = tf.summary.merge(summaries)
    # 3.12 saver
    saver = tf.train.Saver(max_to_keep=saver_maxkeep)
    # 3.13 init all variables
    sess.run(tf.global_variables_initializer())

    restore_saver = tf.train.Saver()
    restore_saver.restore(sess,  ckpt_path + '/InsightFace_iter_'+'90000'+'.ckpt')
    # 4 begin iteration
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    log_file_path = log_file_path + '/train' + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log'
    log_file = open(log_file_path, 'w')
    # 4 begin iteration
    count = 0
    total_accuracy = {}

    for i in range(epoch):
        sess.run(iterator.initializer)
        while True:
            try:
                images_train, labels_train = sess.run(next_element)
                feed_dict = {images: images_train, labels: labels_train, dropout_rate: 0.4}
                feed_dict.update(net.all_drop)
                start = time.time()
                _, total_loss_val, inference_loss_val, wd_loss_val, _, acc_val = \
                    sess.run([train_op, total_loss, inference_loss, wd_loss, inc_op, acc],
                              feed_dict=feed_dict,
                              options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
                end = time.time()
                pre_sec = batch_size/(end - start)
                # print training information
                if count > 0 and count % show_info_interval == 0:
                    print('epoch %d, total_step %d, total loss is %.2f , inference loss is %.2f, weight deacy '
                          'loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                          (i, count, total_loss_val, inference_loss_val, wd_loss_val, acc_val, pre_sec))
                count += 1

                # save summary
                if count > 0 and count % summary_interval == 0:
                    feed_dict = {images: images_train, labels: labels_train, dropout_rate: 0.4}
                    feed_dict.update(net.all_drop)
                    summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    summary.add_summary(summary_op_val, count)

                # save ckpt files
                if count > 0 and count % ckpt_interval == 0:
                    filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                    filename = os.path.join(ckpt_path, filename)
                    saver.save(sess, filename)

                # validate
                if count > 0 and count % validate_interval == 0:
                    feed_dict_test ={dropout_rate: 1.0}
                    feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
                    results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess,
                             embedding_tensor=embedding_tensor, batch_size=batch_size, feed_dict=feed_dict_test,
                             input_placeholder=images)
                    print('test accuracy is: ', str(results[0]))
                    total_accuracy[str(count)] = results[0]
                    log_file.write('########'*10+'\n')
                    log_file.write(','.join(list(total_accuracy.keys())) + '\n')
                    log_file.write(','.join([str(val) for val in list(total_accuracy.values())])+'\n')
                    log_file.flush()
                    if max(results) > 0.996:
                        print('best accuracy is %.5f' % max(results))
                        filename = 'InsightFace_iter_best_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(ckpt_path, filename)
                        saver.save(sess, filename)
                        log_file.write('######Best Accuracy######'+'\n')
                        log_file.write(str(max(results))+'\n')
                        log_file.write(filename+'\n')

                        log_file.flush()
            except tf.errors.OutOfRangeError:
                print("End of epoch %d" % i)
                break
    log_file.close()
    log_file.write('\n')
