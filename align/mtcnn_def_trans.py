from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from align.detect_face_mdoel import *
from os.path import join as pjoin
import time
from PIL import ImageFont, ImageDraw, Image
import math
import os
from skimage import transform as trans
import cv2

# fontpath = "/data/sunruina/face_rg/face_rg_server" + "/data_pro/wryh.ttf"  # 32为字体大小

exe_path = os.path.abspath(__file__)
mark_color = (225, 209, 0)


# 原理关键点 http://www.sfinst.com/?p=1683


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def read_img(person_dir, f):
    img = cv2.imread(pjoin(person_dir, f))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 判断数组维度
    if gray.ndim == 2:
        img = to_rgb(gray)
    return img


def load_data(data_dir):
    data = {}
    pics_ctr = 0
    for guy in os.listdir(data_dir):
        person_dir = pjoin(data_dir, guy)
        curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]
        # 存储每一类人的文件夹内所有图片
        data[guy] = curr_pics
    return data


def embs_toget_names(detect_face_embs_i, known_names_i, known_embs_i):
    L2_dis = np.linalg.norm(detect_face_embs_i - known_embs_i, axis=1)
    is_known = 0
    if min(L2_dis) < 0.55:
        loc_similar_most = np.where(L2_dis == min(L2_dis))
        is_known = 1
        return known_names_i[loc_similar_most][0], is_known
    else:
        return '未知的同学', is_known


def turn_face(b_boxes, points5):
    # bounding_boxes=[[x1,y1,x2,y2,score], [x1,y1,x2,y2,score]]，points_5 = [[x1左眼，x2左眼], [x1右眼，x2右眼],
    # [x1鼻子，x2鼻子], [x左嘴角，..], [x右嘴角, ..], [ y左眼，..], [y右眼，..], [y鼻子，..], [y左嘴角，..], [y右嘴角, ..]]
    # 初始化存储变量
    b_boxes_new, points5_new = b_boxes, points5

    # 关键点图像坐标转化为直角坐标系坐标，即y*(-1)
    points5[5:] = points5[5:] * (-1)

    # 迭代每一张脸
    for fi in range(len(b_boxes)):

        # print('b_boxes[fi], points5 >> fi')
        # print(b_boxes[fi])
        for pi in range(10):
            # print(points5[pi][fi])
            L = np.asarray([points5[0][fi], points5[5][fi]])
            R = np.asarray([points5[1][fi], points5[6][fi]])
            A_v = R - L
            X_v = np.asarray([1, 0])
            cos_a = np.dot(A_v, X_v.T) / (np.linalg.norm(A_v) * np.linalg.norm(X_v))
            sin_a = np.power(1 - np.power(cos_a, 2), 0.5)

            # box图像坐标系转换为直角坐标系
            b_boxes[fi][1] = b_boxes[fi][1] * (-1)
            b_boxes[fi][3] = b_boxes[fi][3] * (-1)
            P1x = b_boxes[fi][0]
            P1y = b_boxes[fi][1]
            P2x = b_boxes[fi][2]
            P2y = b_boxes[fi][3]
            Cx = b_boxes[fi][0] + (b_boxes[fi][2] - b_boxes[fi][0]) / 2
            Cy = b_boxes[fi][1] + (b_boxes[fi][3] - b_boxes[fi][1]) / 2

            P1_xnew = (P1x - Cx) * cos_a - (P1y - Cy) * sin_a + Cx
            P1_ynew = (P1x - Cx) * sin_a + (P1y - Cy) * cos_a + Cx

            P2_xnew = (P2x - Cx) * cos_a - (P2y - Cy) * sin_a + Cx
            P2_ynew = (P2x - Cx) * sin_a + (P2y - Cy) * cos_a + Cx

            b_boxes_new[fi][0] = P1_xnew
            b_boxes_new[fi][1] = P1_ynew
            b_boxes_new[fi][2] = P2_xnew
            b_boxes_new[fi][3] = P2_ynew

            # 计算原始里面，关键点距离原始左上角的距离
            # b_boes_new进行旋转成正型，
            # xin的dets进行cv2旋转取角度值a时， cos_a = 根号2/2，右眼在上，cos值取小的45度，左眼在上cos值取大的315度

    return b_boxes_new, points5_new


def hisEqulColor2(img):
    # 彩色图片自适应均衡化https://blog.csdn.net/Ibelievesunshine/article/details/95220075
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)

    # 以下代码详细注释见官网：
    # https://docs.opencv.org/4.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])

    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def hisEqulColor1(img):
    # 彩色图像全局直方图均衡化 https://blog.csdn.net/Ibelievesunshine/article/details/95220075
    # 将RGB图像转换到YCrCb空间中
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # 将YCrCb图像通道分离
    channels = cv2.split(ycrcb)
    # 对第1个通道即亮度通道进行全局直方图均衡化并保存
    cv2.equalizeHist(channels[0], channels[0])
    # 将处理后的通道和没有处理的两个通道合并，命名为ycrcb
    cv2.merge(channels, ycrcb)
    # 将YCrCb图像转换回RGB图像
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def load_and_align_data(image, det_4para, minsize=100, threshold=[0.65, 0.7, 0.7], factor=0.709,
                        gama_flag=0, anti_flag=1):  # 返回彩图
    # face detection parameters
    # 以下两个阈值调整后，歪脸和遮挡会被过滤掉

    # threshold = [0.6, 0.7, 0.7]  # three steps's threshold 三步的阈值
    # threshold = [0.85, 0.7, 0.7]  # 第一步pnet阈值升高，候选框会比较少
    # threshold = [0.7, 0.7, 0.7]  # 第一步pnet阈值升高，候选框会比较少

    # factor = 0.709  # scale factor 比例因子，越小越快
    # factor = 0.4  # scale factor 比例因子，图像金字塔每次缩小的保留比例，因此factor值越小越快

    # 读取图片
    # cv2.imwrite('b.jpg', image)
    if gama_flag == 1:
        # image = hisEqulColor1(image)
        image = hisEqulColor2(image)
    # cv2.imwrite('a.jpg', image)

    img = to_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    img_size = np.asarray(img.shape)[0:2]
    # 返回边界框数组 （参数分别是输入图片 脸部最小尺寸 三个网络 阈值 ）
    bounding_boxes, points_5 = detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    # bounding_boxes=[[x1,y1,x2,y2,score], [x1,y1,x2,y2,score]]，points_5 = [[x1左眼，x2左眼], [x1右眼，x2右眼], [x1鼻子，x2鼻子], [x左嘴角，..], [x右嘴角, ..], [ y左眼，..], [y右眼，..], [y鼻子，..], [y左嘴角，..], [y右嘴角, ..]]
    if len(bounding_boxes) < 1:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), 0
    else:
        det_f = bounding_boxes

        det_f[:, 0] = np.maximum(det_f[:, 0], 0)
        det_f[:, 1] = np.maximum(det_f[:, 1], 0)
        det_f[:, 2] = np.minimum(det_f[:, 2], img_size[1])
        det_f[:, 3] = np.minimum(det_f[:, 3], img_size[0])
        det_f = det_f.astype(int)
        crop_rg, points_5_crop_rg = align_face(image, bounding_boxes, points_5, det_4para[2], det_4para[3])  # 识别尺寸

        if anti_flag == 1:
            crop_at, points_5_crop_at = align_face(image, bounding_boxes, points_5, det_4para[0], det_4para[1])  # 活体尺寸
        else:
            crop_at, points_5_crop_at = np.asarray([0]), np.asarray([0])
        crop_at, crop_rg = np.stack(crop_at), np.stack(crop_rg)

        return det_f, crop_at, points_5_crop_at, crop_rg, points_5_crop_rg, 1


def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)


def align_face(img, _bbox_raw, _landmark_raw, _image_size, _extend_p):
    """
    used for aligning face

    Parameters:
    ----------
        img : numpy.array
        _bbox : numpy.array shape=(n,1,4)
        _landmark : numpy.array shape=(n,5,2)
        if_align : bool

    Returns:
    -------
        numpy.array

        align_face
    """

    # _landmark_raw = np.asarray(
    #     [['1左眼x', '2左眼x', '3左眼x'],
    #      ['1右眼x', '2右眼x', '3右眼x'],
    #      ['1鼻子x', '2鼻子x', '3鼻子x'],
    #      ['1左嘴x', '2左嘴x', '3左嘴x'],
    #      ['1右嘴x', '2右嘴x', '3右嘴x'],
    #      ['1左眼y', '2左眼y', '3左眼y'],
    #      ['1右眼y', '2右眼y', '3右眼y'],
    #      ['1鼻子y', '2鼻子y', '3鼻子y'],
    #      ['1左嘴y', '2左嘴y', '3左嘴y'],
    #      ['1右嘴y', '2右嘴y', '3右嘴y']])
    # _landmark = np.asarray(
    #     [[['1左眼x', '1左眼y'],
    #       ['1右眼x', '1右眼y'],
    #       ['1鼻子x', '1鼻子y'],
    #       ['1左嘴x', '1左嘴y'],
    #       ['1右嘴x', '1右嘴y'], ],
    #      [['2左眼x', '2左眼y'],
    #       ['2右眼x', '2右眼y'],
    #       ['2鼻子x', '2鼻子y'],
    #       ['2左嘴x', '2左嘴y'],
    #       ['2右嘴x', '2右嘴y'], ],
    #      [['3左眼x', '3左眼y'],
    #       ['3右眼x', '3右眼y'],
    #       ['3鼻子x', '3鼻子y'],
    #       ['3左嘴x', '3左嘴y'],
    #       ['3右嘴x', '3右嘴y'], ]])

    _landmark = np.asarray(np.zeros((len(_landmark_raw[0]), 5, 2)), dtype='str')
    for face_i in range(len(_landmark_raw[0])):
        _landmark[face_i, 0, 0] = _landmark_raw[0, face_i]
        _landmark[face_i, 0, 1] = _landmark_raw[5, face_i]
        _landmark[face_i, 1, 0] = _landmark_raw[1, face_i]
        _landmark[face_i, 1, 1] = _landmark_raw[6, face_i]
        _landmark[face_i, 2, 0] = _landmark_raw[2, face_i]
        _landmark[face_i, 2, 1] = _landmark_raw[7, face_i]
        _landmark[face_i, 3, 0] = _landmark_raw[3, face_i]
        _landmark[face_i, 3, 1] = _landmark_raw[8, face_i]
        _landmark[face_i, 4, 0] = _landmark_raw[4, face_i]
        _landmark[face_i, 4, 1] = _landmark_raw[9, face_i]

    num = np.shape(_bbox_raw)[0]
    warped = np.zeros((num, _image_size, _image_size, 3))
    p5 = np.zeros((num, 5, 2))

    for i in range(num):
        warped[i, :], p5[i, :] = preprocess(img, bbox=_bbox_raw[i], landmark=_landmark[i], image_size=_image_size,
                                            extend_p=_extend_p)

    return warped, p5


def preprocess(img, bbox=None, landmark=None, image_size=112, extend_p=0.0):
    # 只能剪裁出正方形的脸，如果需要 长方形的话需要再外边处理。
    M = None
    # define desire position of landmarks  # 人脸的5个关键点的位置，是固定的
    norm112_112_p5 = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]], dtype=np.float32)

    if image_size != 112:
        norm_any_p5 = norm112_112_p5 * (256 / 112)
    else:
        norm_any_p5 = norm112_112_p5
    # print(norm_any_p5)
    if extend_p != 0.0:
        extend_size = int(image_size * (1 + extend_p))
        norm_any_p5 = norm_any_p5 + int((extend_size - image_size) / 2)  # 如果需要延伸脸周围20%的区域，则在112的标准尺寸上进行放大
    else:
        norm_any_p5 = norm_any_p5
        extend_size = image_size

    # norm_p5 = (norm_p5 - np.min(norm_p5)) /(np.max(norm_p5) - np.min(norm_p5))

    if landmark is not None:  # 如果landmark不为none，就计算出M
        landmark_f = landmark.astype(np.float32)  # 目标关键点，设置一下它的数据类型

        # skimage affine
        tform = trans.SimilarityTransform()  # 引用 class SimilarityTransform()
        tform.estimate(landmark_f[[0, 1], :], norm_any_p5[[0, 1], :])  # 从一组对应的点估计转换,随便两个点就可以，3个点5个点都可以
        M = tform.params[0:2, :]  # 得到(3, 3) 的齐次变换矩阵

    #         #cv2 affine , worse than skimage
    #         src = src[0:3,:]
    #         dst = dst[0:3,:]
    #         M = cv2.getAffineTransform(dst,src)

    if M is None:

        ret = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        ret = cv2.resize(ret, (image_size, image_size))
        landmark = np.asarray(norm112_112_p5, dtype=int)
        return ret, landmark
    else:  # do align using landmark
        warped = cv2.warpAffine(img, M, (extend_size, extend_size), borderValue=0.0)
        move_dictance = int((112 * extend_p) / 2)
        landmark = np.asarray(norm112_112_p5 + move_dictance, dtype=int)

        warped = cv2.resize(warped, (image_size, image_size))
        landmark = np.asarray(landmark * (1 / (1 + extend_p)), dtype=int)
        return warped, landmark


def mark_face_points(points_lst, f_pics):
    # [[x1左眼，x2左眼], [x1右眼，x2右眼], [x1鼻子，x2鼻子], [x左嘴角，..], [x右嘴角, ..], [ y左眼，..], [y右眼，..], [y鼻子，..], [y左嘴角，..], [y右嘴角, ..]]
    cv2.line(f_pics, (points_lst[0][0], points_lst[5][0]), (points_lst[0][0], points_lst[5][0]), mark_color, 2)
    cv2.line(f_pics, (points_lst[1][0], points_lst[6][0]), (points_lst[1][0], points_lst[6][0]), mark_color, 2)
    cv2.line(f_pics, (points_lst[2][0], points_lst[7][0]), (points_lst[2][0], points_lst[7][0]), mark_color, 2)
    cv2.line(f_pics, (points_lst[3][0], points_lst[8][0]), (points_lst[3][0], points_lst[8][0]), mark_color, 2)
    cv2.line(f_pics, (points_lst[4][0], points_lst[9][0]), (points_lst[4][0], points_lst[9][0]), mark_color, 2)

    return f_pics


# 创建mtcnn网络，并加载参数
print('Creating networks and loading parameters')

# gpu设置
gpu_config = tf.ConfigProto()
gpu_config.allow_soft_placement = True
gpu_config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

with tf.Graph().as_default():
    sess = tf.Session(config=gpu_config)
    with sess.as_default():
        pnet, rnet, onet = create_mtcnn(sess, None)

if __name__ == '__main__':
    '''单张图片'''
    # p_path = '/Users/finup/Desktop/rg/face_rg_server_new/data_pro/054258-许杉-公关部.JPG'
    # f_pic = cv2.imread(p_path)
    # det_para = [256, 0.2, 256, 0.0]  # 活体尺寸和延申%，识别尺寸和延伸%
    # dets, crop_images_at, point5_at,  crop_images_rg, point5_rg, align_flag = load_and_align_data(f_pic, det_para, minsize=90)
    # cv2.imwrite('/Users/finup/Desktop/rg/face_rg_server_new/data_pro/陈荟羽-凡信-cropat2.jpg', crop_images_at[0])
    # cv2.imwrite('/Users/finup/Desktop/rg/face_rg_server_new/data_pro/陈荟羽-凡信-croprg2.jpg', crop_images_rg[0])

    '''单层目录'''
    # st = time.time()
    # st_i = st
    # pics_path = '/Users/finup/Desktop/rg/face_rg_server/data_pro/孙瑞娜/'
    # # pics_path = '/Users/finup/Desktop/rg/rg_game/data/Test_Data/'
    # print('pic reading %s' % pics_path)
    # if os.path.isdir(pics_path):
    #     pics_name = list(os.listdir(pics_path))
    #     if '.DS_Store' in pics_name:  # 去掉不为文件夹格式的mac os系统文件
    #         pics_name.remove('.DS_Store')
    #     if 'all_dct.pkl' in pics_name:  # 去掉不为文件夹格式的mac os系统文件
    #         pics_name.remove('all_dct.pkl')
    #     if 'submission_template_s.csv' in pics_name:  # 去掉不为文件夹格式的mac os系统文件
    #         pics_name.remove('submission_template_s.csv')
    #     if 'submission_template.csv' in pics_name:  # 去掉不为文件夹格式的mac os系统文件
    #         pics_name.remove('submission_template.csv')
    #     if 'jumpjump_results.csv' in pics_name:  # 去掉不为文件夹格式的mac os系统文件
    #         pics_name.remove('jumpjump_results.csv')
    # else:
    #     pics_name = [pics_path]
    # print(pics_name)
    #
    # trans_path = pics_path[0:-1] + '_trans/'
    # try:
    #     os.mkdir(trans_path)
    # except:
    #     pass
    # not_found_n = 0
    # for i in range(len(pics_name)):
    #     f_pic = cv2.imread(pics_path + pics_name[i])
    #     dets, crop_images, point5 = load_and_align_data(f_pic, 112, trans_flag=1,
    #                                                        minsize=100, threshold=[0.6, 0.7, 0.7],
    #                                                        factor=0.709)  # minsize 和 threshold 得配合着调节，相互影响，minsize小 threshold也总识别不出来
    #     if len(crop_images) != 0:
    #         cv2.imwrite(trans_path + pics_name[i], crop_images[0])
    #     else:
    #         print(trans_path + pics_name[i], 'not found! ')
    #         not_found_n += 1
    #         cv2.imwrite(trans_path + pics_name[i], f_pic)
    #     if i % 100 == 0:
    #         et_i = time.time()
    #         print('finish:', i, np.round(i / len(pics_name), 2), et_i - st_i, not_found_n)
    #         st_i = et_i
    #
    # print('finish time:', time.time() - st)

    '''两级目录 大量数据人脸数据集'''
    st = time.time()

    # pics_path = '/Users/finup/Desktop/rg/train_data/train_celebrity/celebrity_sample'
    pics_path = '/data/sunruina/face_recognition/data_set/ms_celeb_arcpaper_tfrecords/train_data/train_celebrity/celebrity_sample_246'

    image_size = (112, 112)
    print('pic reading %s' % pics_path)
    if os.path.isdir(pics_path):
        paths = list(os.listdir(pics_path))
        if '.DS_Store' in paths:  # 去掉不为文件夹格式的mac os系统文件
            paths.remove('.DS_Store')
    else:
        paths = [pics_path]

    try:
        os.mkdir(pics_path + '_trans/')
        os.mkdir(pics_path + '_trans_no/')
    except:
        pass

    n_i, mtcnn_1, mtcnn_0 = 0, 0, 0
    all_peopleN = len(paths)
    print('all_people:', len(paths))
    for peo_i in range(all_peopleN):
        peo_pics = list(os.listdir(pics_path + '/' + paths[peo_i] + '/'))
        if '.DS_Store' in peo_pics:  # 去掉不为jpg和png格式的mac os系统文件
            peo_pics.remove('.DS_Store')
        peo_pkg = pics_path + '_trans/' + paths[peo_i] + '/'
        try:
            os.mkdir(peo_pkg)
        except:
            pass
        nn = len(peo_pics)
        for i in range(nn):
            n_i += 1
            p_path = pics_path + '/' + paths[peo_i] + '/' + peo_pics[i]
            f_pic = cv2.imread(p_path)
            det_4para = [256, 0.2, 112, 0.0]
            det_f, _, _, crop_images, points_5, det_flag = load_and_align_data(f_pic, det_4para, minsize=100,
                                                                               threshold=[0.65, 0.7, 0.7], factor=0.709,
                                                                               gama_flag=0, anti_flag=0)
            # 获取人脸, 由于frame色彩空间rgb不对应问题，需统一转为灰色图片
            if len(crop_images) != 0:
                to_picpath = peo_pkg + peo_pics[i]
                cv2.imwrite(to_picpath, crop_images[0])
                mtcnn_1 += 1
            else:
                peo_pkg_no = pics_path + '_trans_no/' + paths[peo_i] + '/'
                try:
                    os.mkdir(peo_pkg_no)
                except:
                    pass
                to_picpath_no = peo_pkg_no + peo_pics[i]
                print(p_path, 'not found !')
                cv2.imwrite(to_picpath_no, f_pic)
                mtcnn_0 += 1

        if n_i % 3 == 0:
            print('Finish peo_n: ', peo_i, all_peopleN, np.round(peo_i / all_peopleN, 2), '  Notfound pic_n: ', mtcnn_0)

    print('finish time:', int(time.time() - st))

    # '''两级目录 大量数据身份证'''
    # st = time.time()
    # # pics_path = '/Users/finup/Desktop/rg/face_rg_server/data_pro/dc_marking_1known_ttt'
    # # pics_path = '/Users/finup/Desktop/finup_face_dataset_trans_no_s'
    # pics_path = '/data/sunruina/face_rg/finup_face_dataset'
    #
    # image_size = (112, 112)
    # print('pic reading %s' % pics_path)
    # if os.path.isdir(pics_path):
    #     paths = list(os.listdir(pics_path))
    #     if '.DS_Store' in paths:  # 去掉不为文件夹格式的mac os系统文件
    #         paths.remove('.DS_Store')
    # else:
    #     paths = [pics_path]
    #
    # try:
    #     os.mkdir(pics_path + '_trans/')
    #     os.mkdir(pics_path + '_trans_no/')
    # except:
    #     pass
    #
    # n_i, mtcnn_1, mtcnn_0 = 0, 0, 0
    # for peo in paths:
    #     peo_pics = list(os.listdir(pics_path + '/' + peo + '/'))
    #     if '.DS_Store' in peo_pics:  # 去掉不为jpg和png格式的mac os系统文件
    #         peo_pics.remove('.DS_Store')
    #     peo_pkg = pics_path + '_trans/' + peo + '/'
    #     try:
    #         os.mkdir(peo_pkg)
    #     except:
    #         pass
    #     nn = len(peo_pics)
    #     for i in range(nn):
    #         n_i += 1
    #         p_path = pics_path + '/' + peo + '/' + peo_pics[i]
    #         f_pic = cv2.imread(p_path)
    #
    #         dets, crop_images, point5 = load_and_align_data(f_pic, 112, trans_flag=1, minsize=90)
    #         # 获取人脸, 由于frame色彩空间rgb不对应问题，需统一转为灰色图片
    #         if len(crop_images) != 0:
    #             to_picpath = peo_pkg + peo_pics[i]
    #             cv2.imwrite(to_picpath, crop_images[0])
    #             mtcnn_1 += 1
    #         else:
    #             peo_pkg_no = pics_path + '_trans_no/' + peo + '/'
    #             try:
    #                 os.mkdir(peo_pkg_no)
    #             except:
    #                 pass
    #             to_picpath_no = peo_pkg_no + peo_pics[i]
    #             print(p_path, 'not found !')
    #             cv2.imwrite(to_picpath_no, f_pic)
    #             mtcnn_0 += 1
    #
    #         if n_i % 1 == 0:
    #             print('finish n: ', n_i, 'notfound n: ', mtcnn_0)
    #
    # print('finish time:', int(time.time() - st))
