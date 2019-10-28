from skimage import io, transform
import os
import numpy as np

root_dir = 'data/'
img_file = 'ADE_train_00003100.png'

img_name = os.path.join(root_dir, img_file)
image = io.imread(img_name)

print (image.shape)
print (image)
# print ObjectClassMasks


