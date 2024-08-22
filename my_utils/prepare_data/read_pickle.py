import os
import pickle


INPUT_PATH = '/data/hqvo3/codes/LaBo/datasets/GlomCrawl/splits/class2images_{}.p'

for split_name in ['train', 'val', 'test']:
  print(f'--- Read {split_name} pickle file ---')

  objects_list = []
  pickle_path = INPUT_PATH.format(split_name)
  with open(pickle_path, "rb") as openfile:
    while True:
      try:
        objects_list.append(pickle.load(openfile))
      except EOFError:
        break

  print('-> len(objects_list): ', len(objects_list)) # 1
  print('-> type(objects_list[0]): ', type(objects_list[0])) # dictionary of mapping from a class name to list of corresponding files
  for key, value in objects_list[0].items():
    print(key, ' ', value)