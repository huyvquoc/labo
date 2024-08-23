import collections
import os
import pickle
import shutil

import cv2
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit


if __name__ == '__main__':
  INPUT_PATH = '/data/hqvo3/Datasets/histo_cls/7_ESCA'
  OUTPUT_PATH = './datasets/7_ESCA/splits'
  N_SPLITS = 1
  os.makedirs(OUTPUT_PATH, exist_ok=True)

  subset_names = os.listdir(INPUT_PATH)
  subset_names.remove('LICENSE.txt')
  print(subset_names)

  TRAIN_SET = subset_names[:-1]
  TEST_SET = subset_names[-1]
  print(TRAIN_SET)
  print(TEST_SET)

  dict_of_alias_clsname_to_actual_clsname = {
    'ADVENT':'adventitia',
    'LAM_PROP':'lamina propria mucosae',
    'MUSC_MUC':'muscularis mucosae',
    'MUSC_PROP':'muscularis propria',
    'REGR_TU':'regression tissue',
    'SH_MAG':'mucosa gastric',
    'SH_OES':'mucosa oesophagus',
    'SUBMUC':'submucosa',
    'SUB_GL':'submucosal glands',
    'TUMOR':'tumour',
    'ULCUS':'ulceration'
  }
  dict_of_classname_to_filenames = {}

  # === For test set, only Charite subset (CHA FULL) ===
  test_set_path = os.path.join(INPUT_PATH, TEST_SET)
  test_cls_sub_folders =  os.listdir(test_set_path)
  test_cls_sub_folders.sort()
  print(test_cls_sub_folders)

  dict_of_classname_to_filenames['test'] = {}
  for cls_sub_folder in test_cls_sub_folders:
      cls_sub_path = os.path.join(test_set_path, cls_sub_folder)
      list_of_files = os.listdir(cls_sub_path)
      list_of_files.sort()
      print(f'{cls_sub_folder}: {len(list_of_files)} files.')

      actual_cls_name = dict_of_alias_clsname_to_actual_clsname[cls_sub_folder]
      dict_of_classname_to_filenames['test'][actual_cls_name] = [os.path.join(TEST_SET, cls_sub_folder, file_name) for file_name in list_of_files]

  # === For train set, 3 subsets: ['VALSET1_UKK', 'VALSET2_WNS', 'VALSET3_TCGA'] ===
  # TODO: split the official train set into train set and val set
  # get list of image names (relative paths) in X and list of labels in y
  X, y = [], []

  for train_set in TRAIN_SET:
      train_set_path = os.path.join(INPUT_PATH, train_set)
      train_cls_sub_folders = os.listdir(train_set_path)
      train_cls_sub_folders.sort()
      print(f'=== {train_set} ===')
  
      for cls_sub_folder in train_cls_sub_folders:
          cls_sub_path = os.path.join(train_set_path, cls_sub_folder)
          list_of_files = os.listdir(cls_sub_path)
          list_of_files.sort()
          print(f'{cls_sub_folder}: {len(list_of_files)} files.')

          actual_cls_name = dict_of_alias_clsname_to_actual_clsname[cls_sub_folder]
          relative_img_paths = [os.path.join(train_set, cls_sub_folder, file_name) for file_name in list_of_files]

          X.extend(relative_img_paths)
          y.extend([actual_cls_name]*len(relative_img_paths))

  # === StratifiedShuffleSplit ===
  X = np.array(X)
  y = np.array(y)
  print('========== CASE STRATIFIED SPLIT ==========')
  sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=0)
  sss.get_n_splits(X, y)
  dict_of_train_val_indices = {}
  for i, (train_indices, val_indices) in enumerate(sss.split(X, y)):
    print(f'[**] len(train_indices): {len(train_indices)}')
    print(f'[**] len(val_indices): {len(val_indices)}')
    print('-> Total indices: ', len(train_indices) + len(val_indices))
    dict_of_train_val_indices[i] = {}
    dict_of_train_val_indices[i]['train'] = train_indices
    dict_of_train_val_indices[i]['val'] = val_indices

  # === distribution after first step ===
  train_samples = X[dict_of_train_val_indices[0]['train']]
  train_labels = y[dict_of_train_val_indices[0]['train']]
  val_samples = X[dict_of_train_val_indices[0]['val']]
  val_labels = y[dict_of_train_val_indices[0]['val']]

  dict_of_classname_to_filenames['train'] = {}
  dict_of_classname_to_filenames['val'] = {}
  for idx, train_sample in enumerate(train_samples):
    train_label = train_labels[idx]
    if train_label not in dict_of_classname_to_filenames['train']:
       dict_of_classname_to_filenames['train'][train_label] = []
    dict_of_classname_to_filenames['train'][train_label].append(train_sample)
  for idx, val_sample in enumerate(val_samples):
    val_label = val_labels[idx]
    if val_label not in dict_of_classname_to_filenames['val']:
       dict_of_classname_to_filenames['val'][val_label] = []
    dict_of_classname_to_filenames['val'][val_label].append(val_sample)

  # ===== create pickle split file for LaBo =====
  for split_name in ['train', 'val', 'test']:
    # TODO: dict_of_classname_to_filenames[split_name] is a sub dictionary of mapping from a class name to list of filenames
    with open(os.path.join(OUTPUT_PATH, 'class2images_{}.p'.format(split_name)), 'wb') as handle:
      pickle.dump(dict_of_classname_to_filenames[split_name], handle, protocol=pickle.HIGHEST_PROTOCOL)