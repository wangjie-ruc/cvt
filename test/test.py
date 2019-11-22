import torch
import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv
img = cv.imread('test/assets/image.png')
mask = cv.imread('test/assets/mask.png')

data = {'image':img, 'mask':mask}

from cvt.transform import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomResizedCrop

hflip = RandomResizedCrop(size=(320, 320))
data = hflip(data)
plt.imshow(data['image'])
plt.show()
plt.imshow(data['mask'])
plt.show()