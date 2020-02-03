from cvt.transform import Resize, RandomCrop, Sequence

tfms = Sequence([
    Resize([256, 256]),
    RandomCrop([224, 224])
])

import cv2 as cv 
img = cv.imread('test/assets/image.jpg')[:,:,[2,1,0]]
data = {'image':img}
data = tfms(data)
import matplotlib.pyplot as plt
plt.imshow(data['image'])
plt.show()