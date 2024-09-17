import os
import json

INPUT_PATH = './datasets/CUB/concepts/class2concepts.json'
OUTPUT_PATH = './datasets/CUB/concepts_bl4/class2concepts.json'

with open(INPUT_PATH, 'r') as file:
  dict_of_mapping_from_classname_to_concepts = json.load(file)
list_of_categos = list(dict_of_mapping_from_classname_to_concepts.keys())
list_of_categos.sort()

dict_of_classes = {}
for catego_idx, catego in enumerate(list_of_categos):
  # list_of_pseudo_concepts = ["pseudo concept 1", "pseudo concept 2", "pseudo concept 3", "pseudo concept 4", "pseudo concept 5"]
  list_of_pseudo_concepts = []
  for concept_id in range(catego_idx*5 + 1, catego_idx*5 + 5 + 1):
    list_of_pseudo_concepts.append(''.join(['pseudo concept ', str(concept_id)]))
  dict_of_classes[catego] = list_of_pseudo_concepts

with open(OUTPUT_PATH, "w", encoding='utf-8') as outfile:
  json.dump(dict_of_classes, outfile, ensure_ascii=False, indent=4)