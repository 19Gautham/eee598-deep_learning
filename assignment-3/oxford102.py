import torch
import torchvision
import scipy

from collections import defaultdict

import matplotlib.pyplot as plt

# https://pytorch.org/vision/0.12/_modules/torchvision/datasets/flowers102.html
oxford_102 = torchvision.datasets.Flowers102(root="./data", download=True)

print(f"Number of images: {len(oxford_102._labels)}")
print(f"Number of classes: {len(set(oxford_102._labels))}")

#Count dict
class_count = defaultdict(int)
for label in oxford_102._labels:
    class_count[label] += 1

print("Class label")

# Number of images per class
file = open(file = "./data/flowers-102/Oxford-102_Flower_dataset_labels.txt", mode="r")
i = 0
label_dict = {}
for line in file:
    line = line.strip()
    label_dict[i] = line.strip()[1:len(line)-1]
    i +=1

# No of images per class
for label in label_dict.keys():
    print(f"Class: {label_dict[label]}, Count: {class_count[label]}")