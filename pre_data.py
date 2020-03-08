"""
This file does image pre processing. It obtains the correct images based on experiment requirement.
Specifically. Three important aspects of the data:
    file_List: list of wanted file names and directories
    imArray_list: list of list, size n-by-(256*256)), all in numpy array format. Representing segmentation map after class combination
    ref_Map: one-hot encoding of each object classes, based on total number of classes in the wanted images.
"""



import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import os
from skimage import io, transform
from one_hot_helper import covertToOnehot
from one_hot_helper import genRefMap
from one_hot_helper import combineClasses


train = True
by_category = True     # Load data from selected categories

ref_root_dir = "/Users/ericfu/Documents/ANU_Master/COMP8755_Project/dataset/ADEChallengeData2016/"
anno_root_dir = "/Users/ericfu/Documents/ANU_Master/COMP8755_Project/dataset/" \
               "ADEChallengeData2016/annotations/training/"

category = ['beach']      # multiple categories stored in a list
ref_list_name = "sceneCategories.txt"
file_list = []         # segmentation maps to use as training examples
imArray_list = []

ref_list = os.path.join(ref_root_dir, ref_list_name)
f = open(ref_list)

classSet = set()    # what classes that the dataset contains.

# Prepare segmentation maps from specified category of scenes ！！
if by_category:

    line = f.readline()
    while line:
        n, c = line.split(" ")
        c = c[:-1]
        path = os.path.join(anno_root_dir, n + ".png")
        if c in category:
            if train:
                if 'train' in n and io.imread(path).shape == (256 ,256):
                    file_list.append(n + ".png")
            else:
                if "val" in n and io.imread(path).shape == (256 ,256):
                    file_list.append(n + ".png")
        line = f.readline()
    f.close()

if by_category is False:

    line = f.readline()
    while line:
        n, c = line.split(" ")
        if train:
            if 'train' in n:
                file_list.append(n + ".png")
        else:
            if "val" in n:
                file_list.append(n + ".png")
        line = f.readline()
    f.close()

# Prepare the image Arrays and class list.
for file in file_list:
    im = io.imread(os.path.join(anno_root_dir, file))
    imArray = np.asarray(im).reshape(1, -1)[0]
    imArray = combineClasses(imArray)
    imArray_list.append(imArray)
    imClasses = np.unique(imArray)
    print("imClass: ", imClasses)
    for c in imClasses:
        classSet.add(c)

print(classSet)
refMap, num_classes = genRefMap(classSet)
print('num classes: ', num_classes)