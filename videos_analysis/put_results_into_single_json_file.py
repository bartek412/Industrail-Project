import glob
import os
import sys
import json

confusion_matrix_path = sys.argv[1]

files = glob.glob(os.getcwd()+'/**/confusion*.json', recursive = True)

file_path = files[0]

confusion_matrices = []
for file_path in files:
    with open(file_path, 'r') as file:
        data = json.load(file)
    confusion_matrices.append(data)
confusion_mat_file = open(confusion_matrix_path, 'w')
json.dump(confusion_matrices, confusion_mat_file, indent=4)