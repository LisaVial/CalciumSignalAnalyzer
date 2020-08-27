import tifffile as tiff
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from IPython import embed

root = r'G:\Lisa\data\Niko'
for root, sub_folders, files in os.walk(root):
    for file in files:
        new_root = root + '\\'
        tiff_stack = tiff.imread(os.path.join(new_root + file))
        for i in range(tiff_stack.shape[0]):
            outfile = os.path.splitext(file)[0] + '_' + str(i) + '.jpeg'
            img = Image.fromarray(tiff_stack[i])
            img.save(outfile)

# img = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)
#
# sift = cv2.xfeatures2d.SURF_create(25000)
# fig, axs = plt.subplots(1, 1)
# kp, des = sift.detectAndCompute(img, None)
#
# img_surf = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 2)
#
# axs.imshow(img_surf)
# plt.show()