import tifffile
import numpy as np
import matplotlib.pyplot as plt
import cv2
from IPython import embed

tif_path = r'G:\Lisa\data\Niko\schoene_Bilder_2.jpg'
# tif_path = r'G:\Lisa\data\Niko\STD_Cortex 10x 20fps gain 2_MMStack_Defaul_test.jpg'
tif_path_2 = r'G:\Lisa\data\Niko\Cortex 10x 20fps gain 2_MMStack_bright_test.jpg'
tif_path_3 = r'G:\Lisa\data\Niko\Cortex 10x 20fps gain 2_MMStack_intermediate_test.jpg'

# with tifffile.TiffFile(tif_path) as tif:
#     imagej_hyperstack = tif.asarray()
# img = imagej_hyperstack[0]
img = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)
# img_bright = cv2.imread(tif_path_2, cv2.IMREAD_GRAYSCALE)
# img_im = cv2.imread(tif_path_3, cv2.IMREAD_GRAYSCALE)
# alpha = 1.5 # Contrast control (1.0-3.0)
# beta = 0 # Brightness control (0-100)
# adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
# adjusted_bright = cv2.convertScaleAbs(img_bright, alpha=alpha, beta=beta)
# adjusted_im = cv2.convertScaleAbs(img_im, alpha=alpha, beta=beta)
# img_blur = cv2.blur(adjusted, (5, 5))
# img_blur_bright = cv2.blur(adjusted_bright, (5, 5))
# img_blur_im = cv2.blur(adjusted_im, (5, 5))

sift = cv2.xfeatures2d.SURF_create(25000)
fig, axs = plt.subplots(1, 1)
kp, des = sift.detectAndCompute(img, None)
# kp_1, des_1 = sift.detectAndCompute(img_blur, None)
# kp_2, des_2 = sift.detectAndCompute(img_bright, None)
# kp_3, des_3 = sift.detectAndCompute(img_blur_bright, None)
# kp_4, des_4 = sift.detectAndCompute(img_im, None)
# kp_5, des_5 = sift.detectAndCompute(img_blur_im, None)

img_surf = cv2.drawKeypoints(img, kp, None, (255,0,0), 2)
# img1 = cv2.drawKeypoints(img_blur, kp_1, None, (255,0,0), 2)
# img2 = cv2.drawKeypoints(img_bright, kp_2, None, (255,0,0), 2)
# img3 = cv2.drawKeypoints(img_blur_bright, kp_3, None, (255,0,0), 2)
# img4 = cv2.drawKeypoints(img_im, kp_4, None, (255,0,0), 2)
# img5 = cv2.drawKeypoints(img_blur_im, kp_5, None, (255,0,0), 2)
# embed()
# exit()
axs.imshow(img_surf)
# axs[1].imshow(img1)
# axs[0].imshow(img2)
# axs[1].imshow(img3)
# axs[0].imshow(img4)
# axs[1].imshow(img5)
plt.show()