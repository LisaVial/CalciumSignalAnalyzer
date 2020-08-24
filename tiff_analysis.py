import tifffile
import numpy as np
import matplotlib.pyplot as plt
import warnings
import cv2
import matplotlib.gridspec as spec
from IPython import embed


def aniso_filter(img, niter=100, kappa=25, gamma=0.1, step=(1., 1.), option=2, ploton=False):
    """""
    Anisotropic diffusion as flattening filter for images
    """""
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    img = img.astype('float32')
    imgout = img.copy()

    delta_s = np.zeros_like(imgout)
    delta_e = delta_s.copy()
    NS = delta_s.copy()
    EW = delta_s.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    if ploton:
        fig = plt.figure(figsize=(20, 5), num='Anisotropic diffusion')
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(img, interpolation='nearest')
        ih = ax2.imshow(imgout, interpolation='nearest', animated=True)
        ax1.set_title('original image')
        ax2.set_title('Iteration 0')

    for ii in range(niter):
        delta_s[:-1, :] = np.diff(imgout, axis=0)
        delta_e[:, :-1] = np.diff(imgout, axis=1)

        if option == 1:
            gS = np.exp(-(delta_s / kappa) ** 2) / step[0]
            gE = np.exp(-(delta_e / kappa) ** 2) / step[1]
        elif option == 2:
            gS = 1. / (1. + (delta_s / kappa) ** 2) / step[0]
            gE = 1. / (1. + (delta_e / kappa) ** 2) / step[1]

        E = gE * delta_e
        S = gS * delta_s

        NS[:] = S
        EW[:] = E

        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        imgout += gamma * (NS + EW)

        if ploton:
            iterstring = 'Iteration %i' % (ii + 1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()

    return imgout


tif_path = r'G:\Lisa\data\Niko\Cortex 10x 20fps gain 2_MMStack_Default.tif'

# define the vertical filter
vertical_filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

# define the horizontal filter
horizontal_filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

with tifffile.TiffFile(tif_path) as tif:
    fig = plt.figure(figsize=(12, 9))
    g = spec.GridSpec(16, 20)

    # axes to compare different filter types
    ax_1 = fig.add_subplot(g[:8, :4])
    ax_2 = fig.add_subplot(g[:8, 4:8])
    ax_3 = fig.add_subplot(g[:8, 8:12])
    ax_4 = fig.add_subplot(g[:8, 12:16])
    ax_5 = fig.add_subplot(g[:8, 16::])

    # axes to compare edge detection of different filter types
    ax_6 = fig.add_subplot(g[8::, :4])
    ax_7 = fig.add_subplot(g[8::, 4:8])
    ax_8 = fig.add_subplot(g[8::, 8:12])
    ax_9 = fig.add_subplot(g[8::, 12:16])
    ax_10 = fig.add_subplot(g[8::, 16::])

    imagej_hyperstack = tif.asarray()
    ax_1.imshow(imagej_hyperstack[139])
    ax_1.set_title('original')

    img_filt = aniso_filter(imagej_hyperstack[0])
    ax_2.imshow(img_filt)
    ax_2.set_title('anisotropic diffusion')

    img_gauss = cv2.GaussianBlur(imagej_hyperstack[0], (5, 5), 0)
    ax_3.imshow(img_gauss)
    ax_3.set_title('gaussian filter')

    img_med = cv2.medianBlur(imagej_hyperstack[0], 3)
    ax_4.imshow(img_med)
    ax_4.set_title('median filter')

    img_blur = cv2.blur(imagej_hyperstack[0], (5, 5))
    ax_5.imshow(img_blur)
    ax_5.set_title('average filter')
    # plt.show()

    n, m, d = imagej_hyperstack.shape

    # initialize the edges image
    edges_img = imagej_hyperstack[139].copy()
    edges_img_aniso = img_filt.copy()
    edges_img_gauss = img_gauss.copy()
    edges_img_median = img_med.copy()
    edges_img_avg = img_blur.copy()

    for row in range(3, m - 3):
        for col in range(3, d - 3):
            # create little local 3x3 box
            # embed()
            local_pixels = imagej_hyperstack[139][row - 1:row + 2, col - 1:col + 2]
            # embed()
            local_pixels_1 = img_filt[row - 1:row + 2, col - 1:col + 2]
            local_pixels_2 = img_gauss[row - 1:row + 2, col - 1:col + 2]
            local_pixels_3 = img_med[row - 1:row + 2, col - 1:col + 2]
            local_pixels_4 = img_blur[row - 1:row + 2, col - 1:col + 2]

            # apply the vertical filter
            # embed()
            try:
                vertical_transformed_pixels = vertical_filter * local_pixels
                vertical_transformed_pixels_1 = vertical_filter * local_pixels_1
                vertical_transformed_pixels_2 = vertical_filter * local_pixels_2
                vertical_transformed_pixels_3 = vertical_filter * local_pixels_3
                vertical_transformed_pixels_4 = vertical_filter * local_pixels_4
            except ValueError:
                embed()

            # remap the vertical score
            vertical_score = vertical_transformed_pixels.sum() / 4
            vertical_score_1 = vertical_transformed_pixels_1.sum() / 4
            vertical_score_2 = vertical_transformed_pixels_2.sum() / 4
            vertical_score_3 = vertical_transformed_pixels_3.sum() / 4
            vertical_score_4 = vertical_transformed_pixels_4.sum() / 4

            # apply the horizontal filter
            try:
                horizontal_transformed_pixels = horizontal_filter * local_pixels
                horizontal_transformed_pixels_1 = horizontal_filter * local_pixels_1
                horizontal_transformed_pixels_2 = horizontal_filter * local_pixels_2
                horizontal_transformed_pixels_3 = horizontal_filter * local_pixels_3
                horizontal_transformed_pixels_4 = horizontal_filter * local_pixels_4
            except ValueError:
                embed()

            # remap the horizontal score
            horizontal_score = horizontal_transformed_pixels.sum() / 4
            horizontal_score_1 = horizontal_transformed_pixels_1.sum() / 4
            horizontal_score_2 = horizontal_transformed_pixels_2.sum() / 4
            horizontal_score_3 = horizontal_transformed_pixels_3.sum() / 4
            horizontal_score_4 = horizontal_transformed_pixels_4.sum() / 4

            # combine the horizontal and vertical scores into a total edge score
            edge_score = (vertical_score ** 2 + horizontal_score ** 2) ** .5
            edge_score_1 = (vertical_score_1 ** 2 + horizontal_score_1 ** 2) ** .5
            edge_score_2 = (vertical_score_2 ** 2 + horizontal_score_2 ** 2) ** .5
            edge_score_3 = (vertical_score_3 ** 2 + horizontal_score_3 ** 2) ** .5
            edge_score_4 = (vertical_score_4 ** 2 + horizontal_score_4 ** 2) ** .5

            # insert this edge score into the edges image
            edges_img[row, col] = np.asarray(edge_score) * 3
            edges_img_aniso[row, col] = np.asarray(edge_score_1) * 3
            edges_img_gauss[row, col] = np.asarray(edge_score_2) * 3
            edges_img_median[row, col] = np.asarray(edge_score_3) * 3
            edges_img_avg[row, col] = np.asarray(edge_score_4) * 3

    # remap the values in the 0-1 range in case they went out of bounds
    # edges_img = edges_img / edges_img.min()
    # edges_img_aniso = edges_img_aniso / edges_img_aniso.min()
    # edges_img_gauss = edges_img_gauss / edges_img_gauss.min()
    # edges_img_median = edges_img_median / edges_img_median.min()
    # edges_img_avg = edges_img_avg / edges_img_avg.min()

    ax_6.imshow(edges_img)
    ax_6.set_title('original edges')

    ax_7.imshow(edges_img_aniso)
    ax_7.set_title('aniso edges')

    ax_8.imshow(edges_img_gauss)
    ax_8.set_title('gauss edges')

    ax_9.imshow(edges_img_median)
    ax_9.set_title('median edges')

    ax_10.imshow(edges_img_avg)
    ax_10.set_title('avg edges')

    plt.savefig('edge_detection_comp_3.png')
    plt.show()
#
#     plt.show()
# #     plt.imshow(imagej_hyperstack[0], origin='upper')
# #     # plt.plot(imagej_hyperstack[0][0])
# #     plt.show()
#
#     # embed()
