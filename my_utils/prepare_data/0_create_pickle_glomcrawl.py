import os
import pickle
import shutil


INPUT_PATH = '/data/hqvo3/Datasets/vlm/ml-scripts/datasets/glom_refine_splits/split0'
NEW_IMG_PATH = '/data/hqvo3/Datasets/vlm/ml-scripts/datasets/glom_refine_labo'
OUTPUT_PATH = './datasets/GlomCrawl/splits'
os.makedirs(NEW_IMG_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

for split_name in ['train', 'val', 'test']:
  dict_of_classnames_and_filenames = {}
  split_path = os.path.join(INPUT_PATH, split_name)
  categos = os.listdir(split_path)
  for catego in categos:
    dict_of_classnames_and_filenames[catego.replace('_', ' ')] = []
    cur_filenames = os.listdir(os.path.join(split_path, catego))
    for cur_filename in cur_filenames:
      dict_of_classnames_and_filenames[catego.replace('_', ' ')].append(''.join([catego, '_', cur_filename.rsplit('.', 1)[0]])) # no extension
      # copy image files to new image path
      src_path = os.path.join(split_path, catego, cur_filename)
      new_filename = ''.join([catego, '_', cur_filename])
      dst_path = os.path.join(NEW_IMG_PATH, new_filename)
      # shutil.copy(src_path, dst_path)

  with open(os.path.join(OUTPUT_PATH, 'class2images_{}.p'.format(split_name)), 'wb') as handle:
    pickle.dump(dict_of_classnames_and_filenames, handle, protocol=pickle.HIGHEST_PROTOCOL)
