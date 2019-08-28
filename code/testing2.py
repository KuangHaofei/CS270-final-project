import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import utils


def estimation_rotation_scaling(img_r, img_s):
    rows, cols = img_r.shape
    # fourier transfer
    f_r = np.fft.fft2(img_r)
    f_r_shift = np.fft.fftshift(f_r)
    f_r_mag = np.abs(f_r_shift)

    f_s = np.fft.fft2(img_s)
    f_s_shift = np.fft.fftshift(f_s)
    f_s_mag = np.abs(f_s_shift)

    # plt.subplot(221), plt.imshow(np.log(f_r_mag), cmap='gray')
    # plt.title('Spectrum Image R'), plt.xticks([]), plt.yticks([])
    # plt.subplot(222), plt.imshow(np.log(f_s_mag), cmap='gray')
    # plt.title('Spectrum Image S'), plt.xticks([]), plt.yticks([])

    # high pass filter
    H = utils.high_pass_filter(rows, cols)
    f_r_hp = H * f_r_mag
    f_s_hp = H * f_s_mag

    # plt.subplot(223), plt.imshow(f_r_hp, cmap='gray')
    # plt.title('HP Spectrum Image R'), plt.xticks([]), plt.yticks([])
    # plt.subplot(224), plt.imshow(f_s_hp, cmap='gray')
    # plt.title('HP Spectrum Image S'), plt.xticks([]), plt.yticks([])

    # log-polar and cross
    Xin, Yin, base = utils.log_polar_param(f_r_hp)

    lp_r, _ = utils.log_polar(f_r_hp, Xin, Yin, base)
    # lp_r = np.log(np.abs(lp_r))
    cp_r = np.fft.fft2(lp_r)

    lp_s, base = utils.log_polar(f_s_hp, Xin, Yin, base)
    # lp_s = np.log(np.abs(lp_s))
    cp_s = np.fft.fft2(lp_s)

    # plt.figure()
    # plt.subplot(121),plt.imshow(np.log(np.abs(lp_r)).T, cmap='gray')
    # plt.title('Log Polar Magnitude Spectrum Image 1'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(np.log(np.abs(lp_s)).T, cmap='gray')
    # plt.title('Log Polar Magnitude Spectrum Image 2'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # phase shift and cross power spectrum
    phase_shift = np.exp(1j * (np.angle(cp_r) - np.angle(cp_s)))
    cps = np.real(np.fft.ifft2(phase_shift))

    y_max, x_max = np.unravel_index(cps.argmax(), cps.shape)

    theta = 180 * y_max / rows
    if theta > 90:
        theta -= 180

    if x_max > cols / 2:
        scaling = x_max - cols
    else:
        scaling = x_max

    scaling = base ** (scaling / 2)

    return theta, scaling


def estimation_translation(img_r, img_s):
    rows, cols = img_r.shape
    # img_r = cv2.copyMakeBorder(img_r, rows, rows, cols, cols, cv2.BORDER_CONSTANT, value=0)
    # img_s = cv2.copyMakeBorder(img_s, rows, rows, cols, cols, cv2.BORDER_CONSTANT, value=0)

    img_r = np.float32(img_r)
    img_s = np.float32(img_s)

    [x0, y0], _ = cv2.phaseCorrelate(img_r, img_s)

    if abs(x0) > rows or abs(y0) > cols:
        return np.nan, np.nan
    else:
        return x0, y0


#########################################################
# create a transfered image
# translation testing
img_r = cv2.imread('lena.png', 0)
rows, cols = img_r.shape

tx = 105.
ty = -115.
theta = 15
scale = 1.3

M_T = np.float32([[1, 0, tx],
                  [0, 1, ty]])

img_s = cv2.warpAffine(img_r, M_T, (cols, rows))

M_RS = cv2.getRotationMatrix2D((cols/2, rows/2), theta, scale)
img_s = cv2.warpAffine(img_s, M_RS, (cols, rows))
# img_r = img_r[256:, :]
# img_s = img_s[256:, :]
# rows, cols = img_r.shape

# img_r = cv2.imread('image1.jpg', 0)
# img_s = cv2.imread('image2.jpg', 0)
#
#
# rows, cols = img_r.shape

plt.figure()
plt.subplot(221), plt.imshow(img_r, cmap='gray')
plt.title('Input Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(img_s, cmap='gray')
plt.title('Input Image 2'), plt.xticks([]), plt.yticks([])

############################################################################
# estimation rotation and scalin
start_time = time.time()
e_theta, e_scale = estimation_rotation_scaling(img_r, img_s)
print("--- %s seconds ---" % (time.time() - start_time))

print(e_theta, e_scale)

M_eRS = cv2.getRotationMatrix2D((cols/2, rows/2), -e_theta, 1/e_scale)

img_t = cv2.warpAffine(img_s, M_eRS, (cols, rows))

############################################################################
# estimation translation
start_time = time.time()
x0, y0 = estimation_translation(img_r, img_t)
print("--- %s seconds ---" % (time.time() - start_time))

print(x0, y0)

M_eT = np.float32([[1, 0, -x0],
                   [0, 1, -y0]])

img_es = cv2.warpAffine(img_t, M_eT, (cols, rows))

# translation result
plt.subplot(223), plt.imshow(img_t, cmap='gray')
plt.title('estimation rotation and scaling'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_es, cmap='gray')
plt.title('estimation result'), plt.xticks([]), plt.yticks([])
# plt.show()

img_combin = np.copy(img_es)

for i in range(rows):
    for j in range(cols):
        if img_es[i, j] == 0:
            img_combin[i,j] = img_r[i, j]

plt.figure()
plt.subplot(131), plt.imshow(img_r, cmap='gray')
plt.title('Input Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_s, cmap='gray')
plt.title('Input Image 2'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_combin, cmap='gray')
plt.title('estimation result: re-rotated -> re-scaled -> re-shifted'), plt.xticks([]), plt.yticks([])
plt.show()