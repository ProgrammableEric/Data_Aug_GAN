from skimage import io, transform
import os
import numpy as np

root_dir = 'data/'
img_file = 'ADE_train_00003100.png'

img_name = os.path.join(root_dir, img_file)
image = io.imread(img_name)

# R = image[:, :, 0]
# G = image[:, :, 1]
# B = image[:, :, 2]

# ObjectClassMasks = (np.uint16(R)/10)*256+np.uint16(G)

print image.shape
print image
# print ObjectClassMasks


