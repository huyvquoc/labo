import os
import json

INPUT_PATH = '/data/hqvo3/Datasets/vlm/ml-scripts/datasets/glom_refine_splits/split0/train'
OUTPUT_PATH = './datasets/GlomCrawl/concepts/class2concepts.json'

list_of_categos = os.listdir(INPUT_PATH)
list_of_categos.sort()
dict_of_classes = {}
for catego in list_of_categos:
  dict_of_classes[catego] = [""]*10

with open(OUTPUT_PATH, "w", encoding='utf-8') as outfile:
  json.dump(dict_of_classes, outfile, ensure_ascii=False, indent=4)