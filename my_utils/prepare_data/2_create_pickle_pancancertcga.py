import collections
import os
import pickle
import shutil

import cv2
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit


def find_nth(haystack: str, needle: str, n: int) -> int:
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

if __name__ == '__main__':
    INPUT_PATH = '/data/hqvo3/Datasets/histo_cls/2_PanCancerTCGA'
    OUTPUT_PATH = './datasets/2_PanCancerTCGA/splits'
    N_SPLITS = 1
    CHOSEN_MAG_LEVEL = '0' # Consider resolution level 0, the UNI paper said that ROIs at around 0.5 micro per pixel
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # TODO: get list of image names (relative paths) in X and list of labels in y
    # TODO: case-stratified and label-stratified
    X, y = [], [] # X is the list of relative paths of wsi folders, y is the list of 
    
    
    # === from IPython file ===
    # 1) get class names
    class_names = os.listdir(INPUT_PATH)
    class_names.remove('LICENSE')
    class_names.sort()
    
    # 2) create dict of mapping from class name and magnification to list of corresponding wsis
    cnt_wsis_of_each_mag = {} # cls names, then mag, then the number of cases in that mag
    all_wsis_of_each_mag = {} # whether same cases in different mags. cls names, then mag, then list of cases in that mag
    
    for cls_idx, class_name in enumerate(class_names):
      cls_name_folder_path = os.path.join(INPUT_PATH, class_name)
      mag_subfolders = os.listdir(cls_name_folder_path)
      mag_subfolders.sort()
    
      cnt_wsis_of_each_mag[class_name] = {}
      all_wsis_of_each_mag[class_name] = {}
    
      for mag_subfolder in mag_subfolders:
          mag_subfolder_path = os.path.join(cls_name_folder_path, mag_subfolder)
          list_of_wsis = os.listdir(mag_subfolder_path)
          list_of_wsis.sort()
    
          cnt_wsis_of_each_mag[class_name][mag_subfolder] = len(list_of_wsis)
          all_wsis_of_each_mag[class_name][mag_subfolder] = list_of_wsis.copy()
    
    # 3) create useful dictionaries to work with
    all_wsis_at_0 = [] # Wsis among different classes have already been unique => Each wsi has only one corresponding class
    all_cases_at_0 = [] # Case ids among different classes have already been unique => Each case id has only one corresponding class
    dict_of_mapping_from_classname_to_caseids = {}
    dict_of_mapping_from_caseid_to_classname = {}
    dict_of_mapping_from_caseid_to_wsis = {}
    dict_of_mapping_from_wsi_to_patchnames = {}
    
    for class_name in class_names:
      cur_cnt_wsis = cnt_wsis_of_each_mag[class_name][CHOSEN_MAG_LEVEL] # only at mag level 0
      cur_all_wsis = all_wsis_of_each_mag[class_name][CHOSEN_MAG_LEVEL] # only at mag level 0
      cur_all_wsis_uniq = list(set(cur_all_wsis))
      assert len(cur_all_wsis) == len(cur_all_wsis_uniq)
      all_wsis_at_0.extend(cur_all_wsis_uniq)
    
      cur_all_cases = [cur_wsi[:find_nth(cur_wsi, '-', 3)] for cur_wsi in cur_all_wsis]
      cur_all_cases_uniq = list(set(cur_all_cases))
      # assert len(cur_all_cases) == len(cur_all_cases_uniq)
      all_cases_at_0.extend(cur_all_cases_uniq)
    
      dict_of_mapping_from_classname_to_caseids[class_name] = cur_all_cases_uniq
      for case_id in cur_all_cases_uniq:
        dict_of_mapping_from_caseid_to_classname[case_id] = class_name
      for idx2, case_id in enumerate(cur_all_cases):
        if case_id not in dict_of_mapping_from_caseid_to_wsis:
          dict_of_mapping_from_caseid_to_wsis[case_id] = []
        dict_of_mapping_from_caseid_to_wsis[case_id].append(cur_all_wsis[idx2])
      for wsi_name in cur_all_wsis:
        abs_wsi_path = os.path.join(INPUT_PATH, class_name, CHOSEN_MAG_LEVEL, wsi_name)
        corr_patchnames = os.listdir(abs_wsi_path)
        corr_patchnames.sort()
        corr_abs_patchnames = [os.path.join(class_name, CHOSEN_MAG_LEVEL, wsi_name, corr_patchname) for corr_patchname in corr_patchnames]
        dict_of_mapping_from_wsi_to_patchnames[wsi_name] = corr_abs_patchnames
      
    # 4)
    total_patches = 0
    for cls_idx, (classname, caseids) in enumerate(dict_of_mapping_from_classname_to_caseids.items()):
        total_cur_wsis = 0
        patch_names = []
        for caseid in caseids:
            cur_wsis = dict_of_mapping_from_caseid_to_wsis[caseid]
            total_cur_wsis += len(cur_wsis)
        
            for cur_wsi in cur_wsis:
                patch_names.extend(dict_of_mapping_from_wsi_to_patchnames[cur_wsi])
        
        total_patches += len(patch_names)
        X.extend(caseids)
        y.extend([classname]*len(caseids))

        print(f'- {cls_idx}) Class {classname} has {len(caseids)} cases, {total_cur_wsis} wsis, {len(patch_names)} patches.')
    print('=> Total patches: ', total_patches)
    X = np.array(X)
    y = np.array(y)
    # === from IPython file ===
    
    # === StratifiedShuffleSplit ===
    print('========== CASE STRATIFIFED SPLIT ==========')
    dict_of_splits = {}
    sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=0)
    sss.get_n_splits(X, y)
    dict_of_train_test_indices = {}
    for i, (train_indices, test_indices) in enumerate(sss.split(X, y)):
      print(f'[**] len(train_indices): {len(train_indices)}')
      print(f'[**] len(test_indices): {len(test_indices)}')
      print('-> Total indices: ', len(train_indices) + len(test_indices))
      dict_of_train_test_indices[i] = {}
      dict_of_train_test_indices[i]['train'] = train_indices
      dict_of_train_test_indices[i]['test'] = test_indices

    # === distribution after first step ===
    print('[*] train_indices: ', dict_of_train_test_indices[0]['train'])
    ori_train_case_samples = X[dict_of_train_test_indices[0]['train']]
    ori_train_case_labels = y[dict_of_train_test_indices[0]['train']]
    test_case_samples = X[dict_of_train_test_indices[0]['test']]
    test_case_labels = y[dict_of_train_test_indices[0]['test']]

    # === Further split train into train and val ===
    dict_of_splits = {}
    sss2 = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=0)
    sss2.get_n_splits(ori_train_case_samples, ori_train_case_labels)
    dict_of_train_val_indices = {}
    for i, (train_indices, val_indices) in enumerate(sss2.split(ori_train_case_samples, ori_train_case_labels)):
      dict_of_train_val_indices[i] = {}
      dict_of_train_val_indices[i]['train'] = train_indices
      dict_of_train_val_indices[i]['val'] = val_indices

    train_case_samples = ori_train_case_samples[dict_of_train_val_indices[0]['train']]
    train_case_labels = ori_train_case_labels[dict_of_train_val_indices[0]['train']]
    val_case_samples = ori_train_case_samples[dict_of_train_val_indices[0]['val']]
    val_case_labels = ori_train_case_labels[dict_of_train_val_indices[0]['val']]

    dict_of_mapping_from_cls_name_to_num_samples_train = {}
    dict_of_mapping_from_cls_name_to_num_samples_val = {}
    dict_of_mapping_from_cls_name_to_num_samples_test = {}
    for train_case_label in train_case_labels:
      if train_case_label not in dict_of_mapping_from_cls_name_to_num_samples_train:
        dict_of_mapping_from_cls_name_to_num_samples_train[train_case_label] = 0
      dict_of_mapping_from_cls_name_to_num_samples_train[train_case_label] += 1
    for val_case_label in val_case_labels:
      if val_case_label not in dict_of_mapping_from_cls_name_to_num_samples_val:
        dict_of_mapping_from_cls_name_to_num_samples_val[val_case_label] = 0
      dict_of_mapping_from_cls_name_to_num_samples_val[val_case_label] += 1
    for test_case_label in test_case_labels:
      if test_case_label not in dict_of_mapping_from_cls_name_to_num_samples_test:
        dict_of_mapping_from_cls_name_to_num_samples_test[test_case_label] = 0
      dict_of_mapping_from_cls_name_to_num_samples_test[test_case_label] += 1

    train_case_ordered_dict = collections.OrderedDict(sorted(dict_of_mapping_from_cls_name_to_num_samples_train.items()))
    val_case_ordered_dict = collections.OrderedDict(sorted(dict_of_mapping_from_cls_name_to_num_samples_val.items()))
    test_case_ordered_dict = collections.OrderedDict(sorted(dict_of_mapping_from_cls_name_to_num_samples_test.items()))
    print('== Train case counter ===')
    check_cnt = 0
    for cls_name, cnt_samples in train_case_ordered_dict.items():
      num_train_indices = len(dict_of_train_val_indices[0]['train'])
      print(f'- {cls_name}: {cnt_samples} with {cnt_samples/num_train_indices}%.')
      check_cnt += cnt_samples
    print(check_cnt)
    print('== Val case counter ===')
    check_cnt = 0
    for cls_name, cnt_samples in val_case_ordered_dict.items():
      num_val_indices = len(dict_of_train_val_indices[0]['val'])
      print(f'- {cls_name}: {cnt_samples} with {cnt_samples/num_val_indices}%.')
      check_cnt += cnt_samples
    print(check_cnt)
    print('== Test case counter ===')
    check_cnt = 0
    for cls_name, cnt_samples in test_case_ordered_dict.items():
      num_test_indices = len(dict_of_train_test_indices[0]['test'])
      print(f'- {cls_name}: {cnt_samples} with {cnt_samples*100/num_test_indices}%.')
      check_cnt += cnt_samples
    print(check_cnt)

    # === check patch-level distribution of classes ===
    train_case_samples
    val_case_samples
    test_case_samples
    train_case_labels
    val_case_labels
    test_case_labels

    dict_of_mapping_from_classname_to_caseids
    dict_of_mapping_from_caseid_to_classname
    dict_of_mapping_from_caseid_to_wsis
    dict_of_mapping_from_wsi_to_patchnames

    dict_of_mapping_from_classname_to_wsis_train = {}
    dict_of_mapping_from_classname_to_wsis_val = {}
    dict_of_mapping_from_classname_to_wsis_test = {}

    dict_of_classname_to_filenames = {}

    dict_of_classname_to_filenames['train'] = {}
    num_train_wsis = 0
    num_train_patches = 0
    for sample_idx, train_caseid in enumerate(train_case_samples):
      assert dict_of_mapping_from_caseid_to_classname[train_caseid] == train_case_labels[sample_idx]
      cur_train_wsis = dict_of_mapping_from_caseid_to_wsis[train_caseid]
      num_train_wsis += len(cur_train_wsis)

      if train_case_labels[sample_idx] not in dict_of_classname_to_filenames['train']:
        dict_of_classname_to_filenames['train'][train_case_labels[sample_idx]] = []
      for cur_train_wsi in cur_train_wsis:
        cur_train_patches = dict_of_mapping_from_wsi_to_patchnames[cur_train_wsi]
        # assert len(cur_train_patches) == 30
        # print('[**] len(cur_train_patches): ', len(cur_train_patches))
        num_train_patches += len(cur_train_patches)

        dict_of_classname_to_filenames['train'][train_case_labels[sample_idx]].extend(dict_of_mapping_from_wsi_to_patchnames[cur_train_wsi])

      if train_case_labels[sample_idx] not in dict_of_mapping_from_classname_to_wsis_train:
        dict_of_mapping_from_classname_to_wsis_train[train_case_labels[sample_idx]] = []
      dict_of_mapping_from_classname_to_wsis_train[train_case_labels[sample_idx]].extend(cur_train_wsis)


    ordered_dict_of_mapping_from_classname_to_wsis_train = collections.OrderedDict(sorted(dict_of_mapping_from_classname_to_wsis_train.items()))
    for classname, wsis in ordered_dict_of_mapping_from_classname_to_wsis_train.items():
      cur_perc = round(len(wsis)*100/num_train_wsis, 2)
      print(f'- {classname}: {len(wsis)} with {cur_perc}%.')

    print(f'=> Pre, the total number of WSIs used for training: {num_train_wsis}')
    print(f'=> Pre, the total number of patches used for training: {num_train_patches}')

    dict_of_classname_to_filenames['val'] = {}
    num_val_wsis = 0
    num_val_patches = 0
    for sample_idx, val_caseid in enumerate(val_case_samples):
      assert dict_of_mapping_from_caseid_to_classname[val_caseid] == val_case_labels[sample_idx]
      cur_val_wsis = dict_of_mapping_from_caseid_to_wsis[val_caseid]
      num_val_wsis += len(cur_val_wsis)

      if val_case_labels[sample_idx] not in dict_of_classname_to_filenames['val']:
        dict_of_classname_to_filenames['val'][val_case_labels[sample_idx]] = []
      for cur_val_wsi in cur_val_wsis:
        cur_val_patches = dict_of_mapping_from_wsi_to_patchnames[cur_val_wsi]
        # assert len(cur_val_patches) == 30
        # print('[**] len(cur_val_patches): ', len(cur_val_patches))
        num_val_patches += len(cur_val_patches)

        dict_of_classname_to_filenames['val'][val_case_labels[sample_idx]].extend(dict_of_mapping_from_wsi_to_patchnames[cur_val_wsi])

      if val_case_labels[sample_idx] not in dict_of_mapping_from_classname_to_wsis_val:
        dict_of_mapping_from_classname_to_wsis_val[val_case_labels[sample_idx]] = []
      dict_of_mapping_from_classname_to_wsis_val[val_case_labels[sample_idx]].extend(cur_val_wsis)

    ordered_dict_of_mapping_from_classname_to_wsis_val = collections.OrderedDict(sorted(dict_of_mapping_from_classname_to_wsis_val.items()))
    for classname, wsis in ordered_dict_of_mapping_from_classname_to_wsis_val.items():
      cur_perc = round(len(wsis)*100/num_val_wsis, 2)
      print(f'- {classname}: {len(wsis)} with {cur_perc}%.')

    print(f'=> Pre, the total number of WSIs used for val: {num_val_wsis}')
    print(f'=> Pre, the total number of patches used for val: {num_val_patches}')

    dict_of_classname_to_filenames['test'] = {}
    num_test_wsis = 0
    num_test_patches = 0
    for sample_idx, test_caseid in enumerate(test_case_samples):
      assert dict_of_mapping_from_caseid_to_classname[test_caseid] == test_case_labels[sample_idx]
      cur_test_wsis = dict_of_mapping_from_caseid_to_wsis[test_caseid]
      num_test_wsis += len(cur_test_wsis)

      if test_case_labels[sample_idx] not in dict_of_classname_to_filenames['test']:
        dict_of_classname_to_filenames['test'][test_case_labels[sample_idx]] = []
      for cur_test_wsi in cur_test_wsis:
        cur_test_patches = dict_of_mapping_from_wsi_to_patchnames[cur_test_wsi]
        # assert len(cur_test_patches) == 30
        # print('[**] len(cur_test_patches): ', len(cur_test_patches))
        num_test_patches += len(cur_test_patches)

        dict_of_classname_to_filenames['test'][test_case_labels[sample_idx]].extend(dict_of_mapping_from_wsi_to_patchnames[cur_test_wsi])

      if test_case_labels[sample_idx] not in dict_of_mapping_from_classname_to_wsis_test:
        dict_of_mapping_from_classname_to_wsis_test[test_case_labels[sample_idx]] = []
      dict_of_mapping_from_classname_to_wsis_test[test_case_labels[sample_idx]].extend(cur_test_wsis)

    ordered_dict_of_mapping_from_classname_to_wsis_test = collections.OrderedDict(sorted(dict_of_mapping_from_classname_to_wsis_test.items()))
    for classname, wsis in ordered_dict_of_mapping_from_classname_to_wsis_test.items():
      cur_perc = round(len(wsis)*100/num_test_wsis, 2)
      print(f'- {classname}: {len(wsis)} with {cur_perc}%.')

    print(f'=> Pre, the total number of WSIs used for testing: {num_test_wsis}')
    print(f'=> Pre, the total number of patches used for testing: {num_test_patches}')


    
    # ===== create pickle split file for LaBo =====
    
    for split_name in ['train', 'val', 'test']:
      # TODO: dict_of_classname_to_filenames[split_name] is a sub dictionary of mapping from a class name to list of filenames
      with open(os.path.join(OUTPUT_PATH, 'class2images_{}.p'.format(split_name)), 'wb') as handle:
        pickle.dump(dict_of_classname_to_filenames[split_name], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # TODO: Following UNI, we do case-stratified and then approximately label-stratified
    # 1st step: consider each case as one sample and try to preserve the percentage of samples for each class
    # 2nd step: consider each patch as one sample and try exchanging cases between splits so that the percentage of samples (now is patches) is preserved. Also guarantee that wsis that are already in one split will not appear in the other splits
    # Because for each wsi, there will be exactly 30 patches (no, there is not exactly 30 patches, can be 40 patches or 60 patches), so balancing the number of wsis for each class will also lead to the balance of patches