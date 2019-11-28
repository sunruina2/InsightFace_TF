import tensorflow as tf
import tensorlayer as tl
import argparse
from data.mx2tfrecords import raw_parse_function, folder_parse_function
import os
from nets.L_Resnet_E_IR_MGPU import get_resnet
from losses.face_losses import arcface_loss
import time
from data.eval_data_reader import load_bin
from verification import ver_test


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--net_depth', default=100, help='resnet depth, default is 50')
    parser.add_argument('--epoch', default=100000, help='epoch to train the network')
    parser.add_argument('--batch_size', default=64, help='batch size to train network')
    parser.add_argument('--lr_steps', default=[40000, 60000, 80000], help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    parser.add_argument('--weight_deacy', default=5e-4, help='learning alg momentum')
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['lfw', 'cfp_fp'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--num_output', default=85164, help='the image size')
    parser.add_argument('--tfrecords_file_path', default='./datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=100000, help='tf dataset api buffer size')  # MGPU 变大*10
    parser.add_argument('--log_device_mapping', default=False,
                        help='show device placement log')  # MGPU 删掉了log_file_path参数
    parser.add_argument('--summary_interval', default=300, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=5000, help='intervals to save ckpt file')  # MGPU 变小/2
    parser.add_argument('--validate_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=20, help='intervals to show information')
    parser.add_argument('--num_gpus', default=[0, 1], help='the num of gpus')  # MGPU
    parser.add_argument('--tower_name', default='tower', help='tower name')  # MGPU
    args = parser.parse_args()
    return args


#  MGPU
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

    # 1. define global parameters
    batch_size = 200  # batch size to train network
    buffer_size = 100000  # tf dataset api buffer size ?
    # buffer_size = 2000  # tf dataset api buffer size ?

    lr_steps = [40000, 60000, 80000, 100000]  # learning rate to train network
    lr_values = [0.005, 0.001, 0.0005, 0.0003, 0.0001]  # learning rate to train network
    # lr_values = [0.0025, 0.0005, 0.00025, 0.00015, 0.00005]  # learning rate to train network
    loss_s = 64
    loss_m = 0.5

    num_output, continue_train_flag, start_count = 179721, 0, 0  # the image size
    tfrecords_file_path = '../train_data/ms1_asiancele.tfrecords'  # path to the output of tfrecords file path
    # tfrecords_file_path = '../train_data/Asian.tfrecords'  # path to the output of tfrecords file path
    pretrain_ckpt_path = '../insight_out/1030_auroua_out/mgpu_res/ckpt/InsightFace_iter_' + str(start_count) + '.ckpt'

    out_dt = '1128'
    summary_path = '../insight_out/' + out_dt + '_ms1assian_s160/mgpu_res/summary'  # the summary file save path
    ckpt_path = '../insight_out/' + out_dt + '_ms1assian_s160/mgpu_res/ckpt'  # the ckpt file save path
    ckpt_count_interval = 50000  # intervals to save ckpt file  # MGPU 变小/2
    # ckpt_count_interval = 10*(int(906/batch_size)+1)  # intervals to save ckpt file  # MGPU 变小/2

    # 打印关键参数到nohup out中
    key_para = {'batch_size': batch_size, 'buffer_size': buffer_size, 'lr_steps': lr_steps, 'lr_values': lr_values,
                'loss_s':loss_s,'loss_m':loss_m,'num_output': num_output, 'tfrecords_file_path': tfrecords_file_path,
                'continue_train_flag': continue_train_flag, 'start_count': start_count,
                'pretrain_ckpt_path': pretrain_ckpt_path, 'out_dt': out_dt, 'summary_path': summary_path,
                'ckpt_path': ckpt_path, 'ckpt_count_interval': ckpt_count_interval}
    for k, v in key_para.items():
        print(k, '        ', v)

    net_depth = 50  # resnet depth, default is 50
    epoch = 100000  # epoch to train the network
    momentum = 0.9  # learning alg momentum
    weight_deacy = 5e-4  # learning alg momentum
    eval_datasets = ['lfw', 'cplfw', 'agedb_30']  # evluation datasets
    eval_db_path = '../ver_data'  # evluate datasets base path
    image_size = [112, 112]  # the image size
    saver_maxkeep = 100  # tf.train.Saver max keep ckpt files
    log_device_mapping = False  # show device placement log  # MGPU 删掉了log_file_path参数
    summary_interval = 500  # interval to save summary
    validate_interval = 5000  # intervals to save ckpt file
    show_info_interval = 10  # intervals to show information
    num_gpus = [0, 1]  # the num of gpus')  # MGPU
    tower_name = 'tower'  # tower name')  # MGPU

    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    images = tf.placeholder(name='img_inputs', shape=[None, *image_size, 3], dtype=tf.float32)
    images_test = tf.placeholder(name='img_inputs', shape=[None, *image_size, 3], dtype=tf.float32)  # MGPU
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    s_n = tf.to_int64(tf.floor(tf.div(tf.to_float(tf.shape(images)[0]), len(num_gpus))))
    sn_lst = [s_n for i in range(len(num_gpus))]
    sn_lst[-1] = s_n + (tf.to_int64(tf.shape(images)[0]) - tf.to_int64(tf.reduce_sum(sn_lst)))
    images_s = tf.split(images, num_or_size_splits=sn_lst,
                        axis=0)  # MGPU 对image和label根据使用的gpu数量做平均拆分（默认两个gpu运算能力相同，如果gpu运算能力不同，可以自己设定拆分策略）
    labels_s = tf.split(labels, num_or_size_splits=sn_lst,
                        axis=0)  # MGPU 对image和label根据使用的gpu数量做平均拆分（默认两个gpu运算能力相同，如果gpu运算能力不同，可以自己设定拆分策略）
    # 2 prepare train datasets and test datasets by using tensorflow dataset api
    # 2.1 train datasets
    # the image is substracted 127.5 and multiplied 1/128.
    # random flip left right
    dataset = tf.data.TFRecordDataset(tfrecords_file_path)
    if tfrecords_file_path.split('/')[-1] in ['ms1.tfrecords', 'ms1v2.tfrecords']:
        dataset = dataset.map(raw_parse_function)  # map，parse_function函数对每一个图进行处理，bgr位置转换，标准化，随机数据增强
    else:
        dataset = dataset.map(folder_parse_function)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    # 2.2 prepare validate datasets
    ver_list = []
    ver_name_list = []
    for db in eval_datasets:
        print('begin db %s convert.' % db)
        data_set = load_bin(db, image_size, eval_db_path)
        ver_list.append(data_set)
        ver_name_list.append(db)
    # 3. define network, loss, optimize method, learning rate schedule, summary writer, saver
    # 3.1 inference phase
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    # 3.2 define the learning rate schedule
    p = int(512.0 / batch_size)
    lr_steps = [p * val for val in lr_steps]
    print('learning rate steps: ', lr_steps)
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=lr_values,
                                     name='lr_schedule')
    # 3.3 define the optimize method
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum)

    # MGPU-start
    # Calculate the gradients for each model tower.
    tower_grads = []  # 保存来自不同GPU计算出的梯度、loss列表
    tl.layers.set_name_reuse(True)
    loss_dict = {}  # 保存来自不同GPU计算出的梯度、loss列表
    drop_dict = {}
    loss_keys = []
    with tf.variable_scope(tf.get_variable_scope()):
        for iter_gpus in num_gpus:
            with tf.device('/gpu:%d' % iter_gpus):
                with tf.name_scope('%s_%d' % (tower_name, iter_gpus)) as scope:
                    net = get_resnet(images_s[iter_gpus], net_depth, type='ir', w_init=w_init_method, trainable=True,
                                     keep_rate=dropout_rate)
                    logit = arcface_loss(embedding=net.outputs, labels=labels_s[iter_gpus], w_init=w_init_method,
                                         out_num=num_output, s= loss_s, m= loss_m)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()  # 同名变量将会复用，假设现在gpu0上创建了两个变量var0，var1，那么在gpu1上创建计算图的时候，如果还有var0和var1，则默认复用之前gpu0上的创建的那两个值
                    # define the cross entropy
                    inference_loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels_s[iter_gpus]))
                    # define weight deacy losses
                    wd_loss = 0
                    for weights in tl.layers.get_variables_with_name('W_conv2d', True, True):
                        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(weights)
                    for W in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/W', True, True):
                        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(W)
                    for weights in tl.layers.get_variables_with_name('embedding_weights', True, True):
                        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(weights)
                    for gamma in tl.layers.get_variables_with_name('gamma', True, True):
                        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(gamma)
                    for alphas in tl.layers.get_variables_with_name('alphas', True, True):
                        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(alphas)
                    total_loss = inference_loss + wd_loss

                    loss_dict[('inference_loss_%s_%d' % ('gpu', iter_gpus))] = inference_loss
                    loss_keys.append(('inference_loss_%s_%d' % ('gpu', iter_gpus)))
                    loss_dict[('wd_loss_%s_%d' % ('gpu', iter_gpus))] = wd_loss
                    loss_keys.append(('wd_loss_%s_%d' % ('gpu', iter_gpus)))
                    loss_dict[('total_loss_%s_%d' % ('gpu', iter_gpus))] = total_loss
                    loss_keys.append(('total_loss_%s_%d' % ('gpu', iter_gpus)))
                    grads = opt.compute_gradients(total_loss)
                    tower_grads.append(grads)  # 把当前GPU计算出的梯度、loss值append到列表
                    if iter_gpus == 0:
                        test_net = get_resnet(images_test, net_depth, type='ir', w_init=w_init_method, trainable=False,
                                              keep_rate=dropout_rate)
                        embedding_tensor = test_net.outputs
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        pred = tf.nn.softmax(logit)
                        acc = tf.reduce_mean(
                            tf.cast(tf.equal(tf.argmax(pred, axis=1), labels_s[iter_gpus]), dtype=tf.float32))

    grads = average_gradients(tower_grads)  # 计算不同GPU获取的grad、loss的平均值
    # Apply the gradients to adjust the shared variables.
    # MGPU-END
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_mapping)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # summary writer
    summary = tf.summary.FileWriter(summary_path, sess.graph)
    summaries = []
    # add grad histogram op
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # add trainabel variable gradients
    for var in tf.trainable_variables():
        if var is not None:
            summaries.append(tf.summary.histogram(var.op.name, var))
    # add loss summary
    for keys, val in loss_dict.items():
        summaries.append(tf.summary.scalar(keys, val))
    # add learning rate
    summaries.append(tf.summary.scalar('leraning_rate', lr))
    summary_op = tf.summary.merge(summaries)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)  # MGPU 没加 max_to_keep=args.saver_maxkeep ，加了tf.global_variables()
    # init all variables
    sess.run(tf.global_variables_initializer())

    if continue_train_flag == 1:
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(
            exclude=['arcface_loss'])  # 不加载包含arcface_loss的所有变量
        print('restore premodel ...')
        restore_saver = tf.train.Saver(variables_to_restore)  # 继续训练的话，将这两行打开
        restore_saver.restore(sess, pretrain_ckpt_path)
    # begin iteration
    count = start_count
    for i in range(epoch):
        sess.run(iterator.initializer)
        while True:
            try:
                images_train, labels_train = sess.run(next_element)
                feed_dict = {images: images_train, labels: labels_train, dropout_rate: 0.4}
                start = time.time()
                _, _, inference_loss_val_gpu_1, wd_loss_val_gpu_1, total_loss_gpu_1, inference_loss_val_gpu_2, \
                wd_loss_val_gpu_2, total_loss_gpu_2, acc_val = sess.run([train_op, inc_op, loss_dict[loss_keys[0]],
                                                                         loss_dict[loss_keys[1]],
                                                                         loss_dict[loss_keys[2]],
                                                                         loss_dict[loss_keys[3]],
                                                                         loss_dict[loss_keys[4]],
                                                                         loss_dict[loss_keys[5]], acc],
                                                                        feed_dict=feed_dict)  # MGPU Loss不同
                end = time.time()
                pre_sec = batch_size / (end - start)
                # print training information
                if count > 0 and count % show_info_interval == 0:
                    # print('epoch %d, total_step %d, total loss gpu 1 is %.2f , inference loss gpu 1 is %.2f, weight deacy '
                    #       'loss gpu 1 is %.2f, total loss gpu 2 is %.2f , inference loss gpu 2 is %.2f, weight deacy '
                    #       'loss gpu 2 is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                    #       (i, count, total_loss_gpu_1, inference_loss_val_gpu_1, wd_loss_val_gpu_1, total_loss_gpu_2,
                    #        inference_loss_val_gpu_2, wd_loss_val_gpu_2, acc_val, pre_sec))

                    print(time.strftime("%Y%m%d%H%M%S", time.localtime()),
                          ' epoch %d, total_step %d, total loss: [%.2f, %.2f], inference loss: [%.2f, %.2f], weight deacy '
                          'loss: [%.2f, %.2f], training accuracy is %.6f, time %.3f samples/sec' %
                          (i, count, total_loss_gpu_1, total_loss_gpu_2, inference_loss_val_gpu_1,
                           inference_loss_val_gpu_2,
                           wd_loss_val_gpu_1, wd_loss_val_gpu_2, acc_val, pre_sec))
                count += 1

                # save summary
                if count > 0 and count % summary_interval == 0:
                    feed_dict = {images: images_train, labels: labels_train, dropout_rate: 0.4}
                    summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    summary.add_summary(summary_op_val, count)

                # save ckpt files
                if count > 0 and count % ckpt_count_interval == 0:
                    filename = 'InsightFace_iter_'+str(count)+'-'+str(i) + '.ckpt'
                    filename = os.path.join(ckpt_path, filename)
                    saver.save(sess, filename)
                # # validate
                if count >= 0 and count % validate_interval == 0:
                    feed_dict_test = {dropout_rate: 1.0}
                    results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess,
                                       embedding_tensor=embedding_tensor, batch_size=batch_size // len(num_gpus),
                                       feed_dict=feed_dict_test,
                                       input_placeholder=images_test)  # MGPU 增加////len(num_gpus)
                    if max(results) > 0.998:
                        print('best accuracy is %.5f' % max(results))
                        filename = 'InsightFace_iter_best_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(ckpt_path, filename)
                        saver.save(sess, filename)
            except tf.errors.OutOfRangeError:
                print("End of epoch %d" % i)
                break
