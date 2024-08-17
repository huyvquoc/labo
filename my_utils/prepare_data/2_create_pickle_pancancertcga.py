import os
import pickle
import shutil

import cv2
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit


INPUT_PATH = '/data/hqvo3/Datasets/histo_cls/2_PanCancerTCGA'
OUTPUT_PATH = './datasets/2_PanCancerTCGA/splits'
N_SPLITS = 1
os.makedirs(OUTPUT_PATH, exist_ok=True)

# TODO: get list of image names (relative paths) in X and list of labels in y
# TODO: case-stratified and label-stratified
X, y = [], [] # X is the list of relative paths of wsi folders, y is the list of 

class_name_folders = os.listdir(INPUT_PATH)
class_name_folders.remove('LICENSE')
class_name_folders.sort()
for class_name_folder in class_name_folders:
  cls_name_folder_path = os.path.join(INPUT_PATH, class_name_folder)
  mag_subfolders = os.listdir(cls_name_folder_path)
  mag_subfolders.sort()

  for mag_subfolder in mag_subfolders:
      print(f'===== {mag_subfolder} =====')
      mag_subfolder_path = os.path.join(cls_name_folder_path, mag_subfolder)
      list_of_wsis = os.listdir(mag_subfolder_path)
      list_of_wsis.sort()
      for wsi in list_of_wsis:
          print('wsi: ', wsi)
          wsi_subfolder_path = os.path.join(mag_subfolder_path, wsi)
          list_of_imgs = os.listdir(wsi_subfolder_path)
          print('=> Number of patches: ', len(list_of_imgs))
          list_of_imgs.sort()
          img_path = os.path.join(wsi_subfolder_path, list_of_imgs[0])

dict_of_splits = {}
sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=0)
sss.get_n_splits(X, y)
for i, (train_index, test_index) in enumerate(sss.split(X, y)):
  print(f"Fold {i}: ")
  print(f"Train: length={len(train_index)}")
  print(f"Test: length={len(test_index)}")
  dict_of_splits[f'split{i}'] = {}
  dict_of_splits[f'split{i}']['train_indices'] =  train_index
  dict_of_splits[f'split{i}']['test_indices'] =  test_index
  print('=> Total unique indices: ', np.unique(np.concatenate((train_index, test_index), axis=0)).shape)

print('common train indices between split 0 and 1: {}'.format(np.intersect1d(dict_of_splits['split0']['train_indices'], dict_of_splits['split1']['train_indices']).shape))
print('common train indices between split 0 and 2: {}'.format(np.intersect1d(dict_of_splits['split0']['train_indices'], dict_of_splits['split2']['train_indices']).shape))
print('common train indices between split 1 and 2: {}'.format(np.intersect1d(dict_of_splits['split1']['train_indices'], dict_of_splits['split2']['train_indices']).shape))

X_array = np.array(X)
y_array = np.array(y)
for split_id in range(N_SPLITS):
  train_indices = dict_of_splits[f'split{split_id}']['train_indices']
  test_indices = dict_of_splits[f'split{split_id}']['test_indices']

  X_sub_train = X_array[train_indices]
  y_sub_train = y_array[train_indices]

  X_sub_test = X_array[test_indices]
  y_sub_test = y_array[test_indices]

  X_train = []
  y_train = []
  for cur_X, cur_y in zip(X_sub_train, y_sub_train):
    cur_dst_path = os.path.join(OUTPUT_PATH, f'split{split_id}', 'train', cur_y)
    os.makedirs(cur_dst_path, exist_ok=True)
    shutil.copy(cur_X, cur_dst_path)

    cur_file_name = cur_X.rsplit('/', 1)[-1]
    X_train.append(os.path.join(cur_dst_path, cur_file_name))
    y_train.append(cur_y)
  for cur_X, cur_y in zip(X_sub_test, y_sub_test):
    cur_dst_path = os.path.join(OUTPUT_PATH, f'split{split_id}', 'test', cur_y)
    os.makedirs(cur_dst_path, exist_ok=True)
    shutil.copy(cur_X, cur_dst_path)

  # === Create val folders from newly generated train folders ===
  sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
  X_train_array = np.array(X_train)
  y_train_array = np.array(y_train)
  for j, (_, val_index) in enumerate(sss2.split(X_train, y_train)):
    # X_sub_train_2 = X_train_array[train_index]
    # y_sub_train_2 = y_train_array[train_index]
    X_sub_val = X_train_array[val_index]
    y_sub_val = y_train_array[val_index]
    
    # move files to val folders
    for cur_X_val, cur_y_val in zip(X_sub_val, y_sub_val):
      cur_dst_path = os.path.join(OUTPUT_PATH, f'split{split_id}', 'val', cur_y_val)
      os.makedirs(cur_dst_path, exist_ok=True)
      shutil.move(cur_X_val, cur_dst_path)


# ===== create pickle split file for LaBo =====

for split_name in ['train', 'val', 'test']:
  # TODO: dict_of_classname_to_filenames[split_name] is a sub dictionary of mapping from a class name to list of filenames
  with open(os.path.join(OUTPUT_PATH, 'class2images_{}.p'.format(split_name)), 'wb') as handle:
    pickle.dump(dict_of_classname_to_filenames[split_name], handle, protocol=pickle.HIGHEST_PROTOCOL)
