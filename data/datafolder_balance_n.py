import os
import shutil
import numpy as np

# path = '/Users/finup/Desktop/rg/train_data/train_celebrity/celebrity/'
# move_path = '/Users/finup/Desktop/rg/train_data/train_celebrity/celebrity_sample/'
path = '/data/sunruina/face_recognition/data_set/ms_celeb_arcpaper_tfrecords/train_data/train_celebrity/celebrity/'
move_path = '/data/sunruina/face_recognition/data_set/ms_celeb_arcpaper_tfrecords/train_data/train_celebrity/celebrity_sample/'
try:
    os.mkdir(move_path)
except:
    pass
sample_num = 7  ##采样数

class_names = os.listdir(path)
class_names.sort()
print('total people is: ', len(class_names))
pi = 0
for label, class_name in enumerate(class_names):
    pi += 1
    if pi % 50 == 0:
        print(np.round(pi / len(class_names), 2))
    classdir = os.path.join(path, class_name)
    if os.path.isdir(classdir):
        pic_names = os.listdir(classdir)
        pic_names.sort()
        try:
            os.mkdir(os.path.join(move_path, class_name))
        except:
            pass
        for num, pic in enumerate(pic_names):
            oldfile = os.path.join(classdir, pic)
            if os.path.isfile(oldfile):
                if num <= sample_num:
                    newfile = os.path.join(os.path.join(move_path, class_name), pic)
                    shutil.copyfile(oldfile, newfile)
                else:
                    break
