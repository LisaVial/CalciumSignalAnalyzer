from PIL import Image
from IPython import embed
import tifffile as tiff
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


def get_circular_rois_mask(img, kp, plot_on = False):
    non_overlap = []
    overlap_indices = []
    masks = []
    h, w = img.shape
    mask = np.zeros((h, w), dtype=bool)
    for keypoint_index in range(len(kp)):
        if keypoint_index in overlap_indices:
            continue    # skip i
        else:
            non_overlap.append(kp[keypoint_index])
        for next_keypoint_index in range(keypoint_index + 1, len(kp)):
            if cv2.KeyPoint_overlap(kp[keypoint_index], kp[next_keypoint_index]) > 0.0:
                overlap_indices.append(next_keypoint_index)

    for non_overlap_index in range(len(non_overlap)):
        x_center, y_center = int(non_overlap[non_overlap_index].pt[0]), int(non_overlap[non_overlap_index].pt[1])
        radius = 2    # radius
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


root = r'D:\Lisa\data\Niko'
surf = cv2.xfeatures2d.SURF_create(400)
fig, axs = plt.subplots(1, 1)
for root, sub_folders, files in os.walk(root):
    for file in files:
        kps = []
        intensities = []
        new_root = root + '\\'
        tiff_stack = tiff.imread(os.path.join(new_root + file))
        rois_over_time = 
        for i in range(tiff_stack.shape[0]):
            outfile = os.path.splitext(file)[0] + '_' + str(i) + '.png'
            plt.imshow(tiff_stack[i])
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            plt.imsave(new_root + outfile, tiff_stack[i], cmap='gray')
            img = cv2.imread(new_root + outfile, cv2.IMREAD_GRAYSCALE)
            # img.thumbnail(img.size)
            # img.convert('RGB')
            # img.save(new_root + outfile, "JPEG", quality=100)
            kp, des = surf.detectAndCompute(img, None)
            kps += kp
            # embed()
            rois = get_circular_rois_mask(img, kps, plot_on=False)
            print(rois.shape, ': \n', rois)

            h, w = rois.shape
            roi_image = np.zeros((h, w, 3), np.uint8)

            for x in range(h-1):
                for y in range(w-1):
                    if rois[x][y]:
                        roi_image[x][y] = img[x][y]
            plt.imshow(roi_image)
            plt.show()
            # for x_idx, y_idx in zip(np.where(rois==True)[0], np.where(rois==True)[2]):
            #     embed()
            #     intensities = img[rois]
            img_surf = cv2.drawKeypoints(img, kps, None, (255, 0, 0), 2)
            outfile_2 = os.path.splitext(file)[0] + '_surf_' + str(i) + '.png'
            axs.imshow(img_surf)
            plt.imsave(new_root + outfile, tiff_stack[i], cmap='gray')
            plt.show()

# img = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)