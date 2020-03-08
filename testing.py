import numpy as np
from numpy import newaxis

t = np.asarray([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])
print(t.shape)
t2 = t[:, :, newaxis]
print(t2.shape)
t3 = t2.reshape(2, 2, -1)
print(t3.shape)

imArray = t.reshape(1, -1)
print(imArray)
imBack = imArray.reshape(4, 4)
print(imBack)

print(t.reshape(4, 2, 2))
