#-*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import os

train = True
by_category = True # Load data from selected categories

ref_root_dir = "/Users/ericfu/Documents/ANU_Master/COMP8755_Project/dataset/ADEChallengeData2016/"
anno_root_dir = "/Users/ericfu/Documents/ANU_Master/COMP8755_Project/dataset/" \
               "ADEChallengeData2016/annotations/training/"

category = 'beach'
ref_list_name = "sceneCategories.txt"
file_list = []         # segmentation maps to use as training examples

ref_list = os.path.join(ref_root_dir, ref_list_name)
f = open(ref_list)

if by_category:

    line = f.readline()
    while line:
        n, c = line.split(" ")
        c = c[:-1]
        if c == category:
            if train:
                if 'train' in n:
                    file_list.append(n + ".png")
            else:
                if "val" in n:
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


class SegMapDataset (Dataset):
    """ Segmentation map datasets for 1st phase of the network. """

    def __init__(self, file_list, anno_root_dir, transform = None):
        self.file_list = file_list
        self.anno_root_dir = anno_root_dir
        self.transform = transform















