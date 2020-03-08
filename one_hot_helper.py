import torch
import numpy as np


## Map relation to combine multiple similar classes into one.
to_Combine = {
    5: [73, 18],  ## tree <- pulm tree, plant,
    2: [26, ],  ## building <- house
    17: [69, ],  ## mountain <- hill
    95: [62, ]  ## land <- bridge
}



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

    print(refMap)

    return refMap, num_classes


def covertToOnehot(imArray, refMap, cNum, Size):

    """
    This function takes the ioread output, and class reference map to encode the original image
    to a one-hot format based on the class reference map.

    :param imgArray, numpy array of size npixels * 1, representing combine object classes of each pixel
    :param refMap: Class reference map, in the form of { pixel value: class code }
    :return: 2D numpy array of one-hot expression, size numClasses-by-numPixels. e.g, 256*256 image with 10 object classes will
             be represented as 10-by-65536 numpy array
    """
    nPix = len(imArray)

    rtn = np.zeros((cNum, nPix))

    for k, pixel in enumerate(imArray):
        rtn[refMap[pixel]][k] = 1

    rtn = rtn.reshape(cNum, Size, Size)

    return rtn


def combineClasses(imArray):

    '''
    This function takes pre-processed image class list and combine object class based on the rule
    defined in to_Combine. And return the modified class list accordingly.
    :param imClasses: 1-by-n np array
    :return: 1-by-n np array
    '''
    to_Combine = {
        5: [73, 18 ], ## tree <- pulm tree, plant,
        2: [26, ], ## building <- house
        17: [69, ], ## mountain <- hill
        95: [62, ] ## land <- bridge
    }

    for c in range(0, len(imArray)):
        for k in to_Combine:
            if imArray[c] in to_Combine[k]:
                imArray[c] = k

    return imArray

# a = np.asarray([[1,2,3], [3,2,1]])
# print(a)
# print(a.reshape(1, -1)[0])