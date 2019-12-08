import torch
import numpy as np

def genRefMap (classSet):
    cArray = []
    i = 0
    for p in classSet:
        cArray.append(p)
        i += 1
    cArray = np.asarray(cArray)
    num_classes = len(cArray)

    refMap = {}
    for i, element in enumerate(cArray):
        refMap[element] = i

    return refMap, num_classes


def covertToOnehot(img, refMap, num_classes):

    img = np.asarray(img).reshape(1, -1)[0]
    num_ele = len(img)

    rtn = np.zeros((num_ele, num_classes))
    for j, pixel in enumerate(img):
        rtn[j][refMap[pixel]] = 1

    print(rtn.shape)
    print(rtn[30000])

    return rtn

