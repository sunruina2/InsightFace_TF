# -*- coding: utf-8 -*-
import os, requests, math
import multiprocessing
import cv2
import time
import numpy as np


def getURLList():
    '''parse csv file, and return a list after simple-processing.'''
    print('read csv ...')
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split(',') for line in lines]
    return lines[1:]  # remove head line


def multi_threads_download_Img(images_lines, threads=8):
    '''using multiprocessing to download images'''
    pool = multiprocessing.Pool(processes=threads)
    for img_i in range(len(images_lines)):
        pool.apply_async(downloadImage, (img_i, 2))
        if img_i % 10 == 0:
            print(img_i, np.round(img_i / len(images_lines), 2))
    pool.close()
    pool.join()


def single_download_Img(images_lines):
    for img_i in range(len(images_lines)):
        downloadImage(img_i, 2)
        if img_i % 10 == 0:
            print(img_i, np.round(img_i / len(images_lines), 2))


def getCoordinate(rect):
    '''get face coordinates in format of xmin, ymin, xmax, ymax'''
    rect = rect.split(' ')
    rect = [int(p) for p in rect]
    return rect


def getHeightWidth(hw):
    '''get raw height and width of image'''
    hw = hw.split(' ')
    hw = [int(p) for p in hw]
    return hw


def faceCrop(img, xmin, ymin, xmax, ymax, scale_ratio=2):
    '''
    crop face from image, the scale_ratio used to control margin size around face.
    using a margin, when aligning faces you will not lose information of face
    '''
    hmax, wmax, _ = img.shape
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = (xmax - xmin) * scale_ratio
    h = (ymax - ymin) * scale_ratio
    # new xmin, ymin, xmax and ymax
    xmin = x - w / 2
    xmax = x + w / 2
    ymin = y - h / 2
    ymax = y + h / 2
    # 大小修正
    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(wmax, int(xmax))
    ymax = min(hmax, int(ymax))

    face = img[ymin:ymax, xmin:xmax, :]
    return face


def downloadImage(line_i, scale_ratio):
    '''download image and crop face from raw image'''
    line = image_lines[line_i]
    name, index, image, rect, hw, url = line
    image_name = '_'.join([name, image])
    # make dir
    peo_pkg_path = os.path.join(root_dir, name[0].lower(), index)
    if not os.path.exists(peo_pkg_path):
        os.makedirs(peo_pkg_path)
        print('os.makedirsroot_dir, name:', peo_pkg_path)
    # jump downloaded image
    img_path = os.path.join(peo_pkg_path, image_name)
    cropimg_path = img_path.replace('.jpg', '_crop.jpg')
    if os.path.exists(cropimg_path):
        # print('os.path.exists(os.path.join(root_dir, name, image_file))')
        return

    # try download image
    try:
        # print('try', url)
        image_data = requests.get(url)
        # image_data = requests.get(url, headers=headers)
        if not image_data.ok:  # there are some wrong urls
            print('url_bug_with_name:', image_name, 'url', url)
            return
        else:
            # print('get success:', img_path)
            with open(img_path, 'wb') as f:
                f.write(image_data.content)
    except Exception as e:
        print('url bug with file:', img_path, 'url:', url, 'e:', e)
        return

    # check image size
    right_height, right_width = getHeightWidth(hw)
    img = cv2.imread(img_path)
    real_height, real_width, _ = img.shape
    if (right_height != real_height) or (right_width != real_width):
        if abs(right_height / right_width - real_height / real_width) < 0.01:
            # print('若实际url长宽与csv长宽不同时，判断url图的长宽比值和csv长宽比值之差小于0.01的情况下，做resize，否则丢弃')
            img = cv2.resize(img, (right_width, right_height))
        else:
            return  # real image size not equal to record

    # crop face from image
    location = getCoordinate(rect)
    # print('faceCrop according to record')
    face = faceCrop(img, *location, scale_ratio)
    cv2.imwrite(cropimg_path, face)
    # delete temp image file to save disk space
    os.remove(img_path)


if __name__ == '__main__':

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'}
    root_dir = '/Users/finup/Desktop/rg/train_data/1mdb_face/IMDB_crop/'  # dir to save face
    csv_file = '/Users/finup/Desktop/rg/train_data/1mdb_face/IMDb-Face-s.csv'  # csv file contains image information

    try:
        os.mkdir(root_dir)
    except:
        pass

    image_lines = getURLList()
    print('image_lines shape：', len(image_lines))

    # st = time.time()
    # multi_threads_download_Img(image_lines, 4)
    # print((time.time() - st) / 60)

    st = time.time()
    single_download_Img(image_lines)
    print((time.time() - st) / 60)
