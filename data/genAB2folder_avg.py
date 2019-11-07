import os
import shutil

if __name__ == '__main__':
    root_path = '/Users/finup/Desktop/rg/train_data/training_folder/dc_marking_trans_AB/'
    pa = root_path.replace('dc_marking_trans_AB/', 'dc_marking_trans_avg_k/')
    pb = root_path.replace('dc_marking_trans_AB/', 'dc_marking_trans_avg_uk/')
    try:
        os.mkdir(pa)
    except:
        pass
    try:
        os.mkdir(pb)
    except:
        pass

    ids = list(os.listdir(root_path))
    if '.DS_Store' in ids:
        ids.remove('.DS_Store')
    ids.sort()
    print(ids)
    for i in range(len(ids)):
        ids_son = list(os.listdir(root_path + ids[i]))
        if '.DS_Store' in ids_son:
            ids_son.remove('.DS_Store')
        ids_son_ab = [a[0] for a in ids_son]
        if len(ids_son) >= 2 and 'A' in ids_son_ab and 'B' in ids_son_ab:
            if not os.path.exists(pa + ids[i]):
                os.mkdir(pa + ids[i])
            if not os.path.exists(pb + ids[i]):
                os.mkdir(pb + ids[i])

            for j in range(len(ids_son)):
                if j < int(len(ids_son)/2):
                    shutil.copy(root_path + ids[i] + '/' + ids_son[j], pa + ids[i] + '/' + ids_son[j])
                else:
                    shutil.copy(root_path + ids[i] + '/' + ids_son[j], pb + ids[i] + '/' + ids_son[j])
