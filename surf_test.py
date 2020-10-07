import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from IPython import embed


def get_circular_rois_mask(img, kp, plot_on = False):
    non_overlap = []
    overlap_indices = []
    masks = []
    h, w = img.shape[0], img.shape[1]
    mask = np.zeros((h, w), dtype=bool)
    for keypoint_index in range(len(kp)):
        if keypoint_index in overlap_indices:
            continue    # skip i
        else:
            non_overlap.append(kp[keypoint_index])
        for next_keypoint_index in range(keypoint_index + 1, len(kp)):
            if cv2.KeyPoint_overlap(kp[keypoint_index], kp[next_keypoint_index]) > 0.0:
                overlap_indices.append(next_keypoint_index)

    # for non_overlap_index in range(len(non_overlap)):
    for non_overlap_index in range(len(non_overlap)):
        x_center, y_center = int(non_overlap[non_overlap_index].pt[0]), int(non_overlap[non_overlap_index].pt[1])
        radius = 10    # radius
        mask_mask = np.array([])
        for x_offset in range(-radius, radius):
            for y_offset in [0, radius/2, 0, -(radius/2), radius, radius/2, 0, -(radius/2), radius, radius/2, 0,
                             -(radius/2), 0]:
                if x_center + x_offset < w and y_center + y_offset < h:
                    px_idx = [(x_center + x_offset, y_center + y_offset)]
                    px_idx_tuple = np.empty(len(px_idx), dtype=object)
                    px_idx_tuple[:] = px_idx
                    mask_mask = np.append(mask_mask, px_idx)
                    mask_mask = np.asarray([(mask_mask[i], mask_mask[i+1]) for i in range(0, len(mask_mask), 2)
                                            if i + 1 < len(mask_mask)])
        ax_circ = plt.subplot(111)
        ax_circ.imshow(img)
        ax_circ.add_patch(plt.Circle((x_center, y_center), radius=radius, color='red', fill=False))

        for pixel_index in mask_mask:
            mask[int(pixel_index[1]), int(pixel_index[0])] = True
        if plot_on and non_overlap_index == len(non_overlap) - 1:
            plt.show()
    return mask


root = r'G:\Lisa\data\Niko\tiff_stack.png'
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=16000, nOctaves=8)
fig, axs = plt.subplots(1, 1)
# for root, sub_folders, files in os.walk(root):
#     for file in files:
#         if file == 'Cortex 10x 20fps gain 2_MMStack_Default.tif':
kps = []
intensities = []
new_root = root + '\\'
# tiff_stack = cv2.imread(os.path.join(new_root + file), flags=cv2.IMREAD_GRAYSCALE)
# tiff_stack = cv2.imreadmulti(os.path.join(new_root + file), flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
# tiff_stack = cv2.imreadmulti(root, flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
# embed()
# for i in range(len(tiff_stack[1])):
#     print(i)
#     # outfile = os.path.splitext(file)[0] + '_' + str(i) + '.png'
#     # plt.imshow(tiff_stack[i])
#     # ax = plt.gca()
#     # ax.get_xaxis().set_visible(False)
#     # ax.get_yaxis().set_visible(False)
#     # ax.get_xaxis().set_ticks([])
#     # ax.axes.get_yaxis().set_ticks([])
#     # plt.imsave(new_root + outfile, tiff_stack[i], cmap='gray')
#     img = tiff_stack[1][i]
img = cv2.imread(root)
# img.thumbnail(img.size)
# img.convert('RGB')
# img.save(new_root + outfile, "JPEG", quality=100)
# change that.... maybe collect kps and only get circulat rois mask at the end of the loop
kp, des = surf.detectAndCompute(img, None)
# embed()
print('%i rois detected, now drawing them' % len(kp))
rois = get_circular_rois_mask(img, kp)

h, w = rois.shape
roi_image = img

for x in range(h-1):
    for y in range(w-1):
        if rois[x][y]:
            roi_image[x][y] = img[x][y]

# outfile_2 = os.path.splitext(root)[0] + '_surf_' + str(i) + '.png'

plt.imshow(img)
plt.imshow(roi_image)
plt.show()
                    # for x_idx, y_idx in zip(np.where(rois==True)[0], np.where(rois==True)[2]):
                    #     embed()
                    #     intensities = img[rois]
                # img_surf = cv2.drawKeypoints(img, kps, None, (255, 0, 0), 2)
                # axs.imshow(img_surf)
                # plt.imsave(new_root + outfile, tiff_stack[i], cmap='gray')
                # plt.show()

# img = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)