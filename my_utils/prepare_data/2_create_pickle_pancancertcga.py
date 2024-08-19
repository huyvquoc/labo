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
    train_case_samples = X[dict_of_train_test_indices[0]['train']]
    train_case_labels = y[dict_of_train_test_indices[0]['train']]
    test_case_samples = X[dict_of_train_test_indices[0]['test']]
    test_case_labels = y[dict_of_train_test_indices[0]['test']]

    dict_of_mapping_from_cls_name_to_num_samples_train = {}
    dict_of_mapping_from_cls_name_to_num_samples_test = {}
    for train_case_label in train_case_labels:
      if train_case_label not in dict_of_mapping_from_cls_name_to_num_samples_train:
        dict_of_mapping_from_cls_name_to_num_samples_train[train_case_label] = 0
      dict_of_mapping_from_cls_name_to_num_samples_train[train_case_label] += 1
    for test_case_label in test_case_labels:
      if test_case_label not in dict_of_mapping_from_cls_name_to_num_samples_test:
        dict_of_mapping_from_cls_name_to_num_samples_test[test_case_label] = 0
      dict_of_mapping_from_cls_name_to_num_samples_test[test_case_label] += 1

    train_case_ordered_dict = collections.OrderedDict(sorted(dict_of_mapping_from_cls_name_to_num_samples_train.items()))
    test_case_ordered_dict = collections.OrderedDict(sorted(dict_of_mapping_from_cls_name_to_num_samples_test.items()))
    print('== Train case counter ===')
    check_cnt = 0
    for cls_name, cnt_samples in train_case_ordered_dict.items():
      num_train_indices = len(dict_of_train_test_indices[0]['train'])
      print(f'- {cls_name}: {cnt_samples} with {cnt_samples/num_train_indices}%.')
      check_cnt += cnt_samples
    print(check_cnt)
    print('== Test case counter ===')
    check_cnt = 0
    for cls_name, cnt_samples in test_case_ordered_dict.items():
      num_test_indices = len(dict_of_train_test_indices[0]['test'])
      print(f'- {cls_name}: {cnt_samples} with {cnt_samples/num_test_indices}%.')
      check_cnt += cnt_samples
    print(check_cnt)

    
    # ===== create pickle split file for LaBo =====
    
    # for split_name in ['train', 'val', 'test']:
    #   # TODO: dict_of_classname_to_filenames[split_name] is a sub dictionary of mapping from a class name to list of filenames
    #   with open(os.path.join(OUTPUT_PATH, 'class2images_{}.p'.format(split_name)), 'wb') as handle:
    #     pickle.dump(dict_of_classname_to_filenames[split_name], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # TODO: Following UNI, we do case-stratified and then approximately label-stratified
    # 1st step: consider each case as one sample and try to preserve the percentage of samples for each class
    # 2nd step: consider each patch as one sample and try exchanging cases between splits so that the percentage of samples (now is patches) is preserved. Also guarantee that wsis that are already in one split will not appear in the other splits
    # Because for each wsi, there will be exactly 30 patches, so balancing the number of wsis for each class will also lead to the balance of patches