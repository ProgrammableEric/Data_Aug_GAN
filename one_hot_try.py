import torch
import numpy as np

def genRefMap (classSet):

    """
    This function takes the set of unique pixel values representing classes of semantic categories, and
    convert them into a reference map in the form of { pixel value: class code }, with the class code starting
    from 0 to number of classes in the selected dataset.
    :param classSet: Python Set of size num_classes
    :return: Python Map with each class' pixel value as the key, and class code as value.
    """

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

    """
    This function takes the ioread output, and class reference map to encode the original image
    to a one-hot format based on the class reference map.

    :param img: of type 'imageio.core.util.Array'
    :param refMap: Class reference map, in the form of { pixel value: class code }
    :param num_classes: Total number of semantic classes within the dataset
    :return: 3D numpy array, one-hot encoded image of size (H-by-W-by-num_classes)
    """
    h = img.shape[0]
    w = img.shape[1]
    img = np.asarray(img)
    num_ele = h * w

    rtn = np.zeros((num_classes, h, w))
    for j, _ in enumerate(img):
        for k, pixel in enumerate(img[j]):
            rtn[refMap[pixel]][j][k] = 1

    return rtn

