import os
import pickle


INPUT_PATH = '/data/hqvo3/Datasets/vlm/ml-scripts/datasets/glom_refine_splits/split0'
OUTPUT_PATH = './datasets/GlomCrawl/splits'
os.makedirs(OUTPUT_PATH, exist_ok=True)

for split_name in ['train', 'val', 'test']:
  dict_of_classnames_and_filenames = {}
  split_path = os.path.join(INPUT_PATH, split_name)
  categos = os.listdir(split_path)
  for catego in categos:
    dict_of_classnames_and_filenames[catego] = []
    cur_filenames = os.listdir(os.path.join(split_path, catego))
    for cur_filename in cur_filenames:
      dict_of_classnames_and_filenames[catego].append(cur_filename.rsplit('.', 1)[0])

  with open(os.path.join(OUTPUT_PATH, 'class2images_{}.p'.format(split_name)), 'wb') as handle:
    pickle.dump(dict_of_classnames_and_filenames, handle, protocol=pickle.HIGHEST_PROTOCOL)
