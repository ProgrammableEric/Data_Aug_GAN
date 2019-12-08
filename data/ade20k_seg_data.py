#-*- coding: UTF-8 -*-

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
from one_hot_try import covertToOnehot
from one_hot_try import genRefMap

#np.set_printoptions(threshold=np.inf)

train = True
by_category = True     # Load data from selected categories

ref_root_dir = "/Users/ericfu/Documents/ANU_Master/COMP8755_Project/dataset/ADEChallengeData2016/"
anno_root_dir = "/Users/ericfu/Documents/ANU_Master/COMP8755_Project/dataset/" \
               "ADEChallengeData2016/annotations/training/"

category = ['beach']      # multiple categories stored in a list
ref_list_name = "sceneCategories.txt"
file_list = []         # segmentation maps to use as training examples

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

# Prepare the class list.
for file in file_list:
    im = io.imread(os.path.join(anno_root_dir, file))
    imArray = np.asarray(im).reshape(1, -1)
    imClasses = np.unique(imArray)
    for c in imClasses:
        classSet.add(c)

testimg = io.imread(os.path.join(anno_root_dir, file_list[1]))

refMap, num_classes = genRefMap(classSet)
covertToOnehot(testimg, refMap, num_classes)

# class SegMapDataset (Dataset):
#     """ Segmentation map dataset for 1st phase of the network. """
#
#     def __init__(self, file_list, anno_root_dir, transform=None):
#         """
#         Args:
#             :param file_list: list of selected image file names to retrieve
#             :param anno_root_dir: Directory that the samples are stored
#             :param transform: Optional transform to be applied
#             on a sample.
#         """
#         self.file_list = file_list
#         self.anno_root_dir = anno_root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(file_list)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         seg_map_name = os.path.join(self.anno_root_dir, self.file_list[idx])
#         image = io.imread(seg_map_name)
#
#         sample = {'image': image, 'fileName': seg_map_name}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample
#
#
# class Rescale(object):
#     """Rescale the image in a sample to a given size. Always assume that we want
#     a square image input (h=w)
#
#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """
#
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size
#
#     def __call__(self, sample):      # 为了将一个类作为函数调用
#         image, fileName = sample['image'], sample['fileName']
#
#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             # if h > w:
#             #     new_h, new_w = self.output_size * h / w, self.output_size
#             # else:
#             #     new_h, new_w = self.output_size, self.output_size * w / h
#             new_h = self.output_size
#             new_w = self.output_size
#         else:
#             new_h, new_w = self.output_size
#
#         new_h, new_w = int(new_h), int(new_w)
#
#         img = transform.resize(image, (new_h, new_w), preserve_range=True)
#
#         # h and w are swapped for landmarks because for images,
#         # x and y axes are axis 1 and 0 respectively
#
#         return {'image': img, 'fileName': fileName}
#
# myData = SegMapDataset(file_list=file_list, anno_root_dir=anno_root_dir)
#
# k = 0
#
# for i in range(len(myData)):
#     sample = myData[i]
#     print(sample['image'].shape, sample['fileName'])
#
# dataloader = DataLoader(myData, batch_size=4,               # load data in chosen manner
#                             shuffle=True, num_workers=4)
#
# print(myData[3]['image'])
# plt.pause(10)













