import os
import shutil

if __name__ == '__main__':
    # root_a = '/Users/finup/Desktop/rg/train_data/training_folder/dc_marking_1known_trans/'
    # root_b = '/Users/finup/Desktop/rg/train_data/training_folder/dc_marking_trans/'
    # root_c = '/Users/finup/Desktop/rg/train_data/training_folder/dc_marking_trans_AB/'
    root_a = '/data/sunruina/face_rg/training_folder/dc_marking_1known_trans/'
    root_b = '/data/sunruina/face_rg/training_folder/dc_marking_trans/'
    root_c = '/data/sunruina/face_rg/training_folder/dc_marking_trans_AB/'
    if not os.path.exists(root_c):
        os.mkdir(root_c)
    ids_a = list(os.listdir(root_a))
    ids_b = list(os.listdir(root_b))
    if '.DS_Store' in ids_a:
        ids_a.remove('.DS_Store')
    if '.DS_Store' in ids_b:
        ids_b.remove('.DS_Store')
    ids_a.sort()
    ids_b.sort()
    print(ids_a)
    print(ids_b)
    if len(ids_a) == len(ids_b):
        for i in range(len(ids_a)):
            if not os.path.exists(root_c + ids_a[i]):
                os.mkdir(root_c + ids_a[i])
            print(root_a + ids_a[i])
            print(root_b + ids_a[i])
            pics_a = os.listdir(root_a + ids_a[i]+'/')
            pics_b = os.listdir(root_b + ids_a[i]+'/')
            for a in pics_a:
                shutil.copy(root_a + ids_a[i]+'/'+a, root_c + ids_a[i]+'/A_'+a)
            for b in pics_b:
                shutil.copy(root_b + ids_a[i]+'/'+b, root_c + ids_a[i]+'/B_'+b)
