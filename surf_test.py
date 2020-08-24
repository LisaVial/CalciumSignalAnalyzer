import tifffile
import numpy as np
import matplotlib.pyplot as plt
import cv2
from IPython import embed

tif_path = r'G:\Lisa\data\Niko\Cortex 10x 20fps gain 2_MMStack_Default.tif'
#
with tifffile.TiffFile(tif_path) as tif:
    imagej_hyperstack = tif.asarray()
img = imagej_hyperstack[0]
# img = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(img)
plt.show()
sift = cv2.xfeatures2d.SIFT_create()

kp, des = sift.detectAndCompute(img, None)
print(len(kp))
img2 = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
plt.imshow(img2)
plt.show()