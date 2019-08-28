import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio


def phase_correlation(src_1, src_2):
    rows, cols = src_1.shape
    src_ph = cv2.copyMakeBorder(src_1, rows, rows, cols, cols, cv2.BORDER_CONSTANT, value=0)
    dst_ph = cv2.copyMakeBorder(src_2, rows, rows, cols, cols, cv2.BORDER_CONSTANT, value=0)

    src_1_pc = np.float32(src_ph)
    src_2_pc = np.float32(dst_ph)

    [x0, y0], _ = cv2.phaseCorrelate(src_1_pc, src_2_pc)

    if abs(x0) > rows or abs(y0) > cols:
        return np.nan, np.nan
    else:
        return x0, y0


def high_pass_filter(ht, wd):
    eta = np.cos(np.pi * np.linspace(-0.5, 0.5, num=ht)).reshape(1, ht)
    neta = np.cos(np.pi * np.linspace(-0.5, 0.5, num=wd)).reshape(1, wd)

    X = np.dot(eta.T, neta)

    H = (1.0 - X) * (2.0 - X)

    return H


# def phase_correlation(src_1, src_2):
#     rows, cols = src_1.shape
#     x0 = 0
#     y0 = 0
#     # Fourier Transfer
#     f1 = np.fft.fft2(src_1)
#     f2 = np.fft.fft2(src_2)
#
#     # f1_shift = np.fft.fftshift(f1)
#     # f2_shift = np.fft.fftshift(f2)
#
#     R = f1 * np.conj(f2) / np.absolute(f1 * np.conj(f2))
#
#     # r_shift = np.fft.ifftshift(R)
#     r = np.fft.ifft2(R)
#
#     x0, y0 = np.unravel_index(r.argmax(), r.shape)
#
#     x0 = rows - x0
#     y0 = cols - y0
#
#     # if x0 >
#
#
#     return x0, y0


img = cv2.imread('lena.png', 0)
rows, cols = img.shape

#########################################################
# translation testing
tx = 105.
ty = -115.

M = np.float32([[1, 0, tx],
                [0, 1, ty]])

dst = cv2.warpAffine(img, M, (cols, rows))

x0, y0 = phase_correlation(src_1=img, src_2=dst)
ret = np.array([x0, y0])
print(np.round(ret))

# cv2.namedWindow('img', flags=cv2.WINDOW_NORMAL)
# cv2.namedWindow('dst', flags=cv2.WINDOW_NORMAL)
# cv2.imshow('img', img)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)

# #########################################################
# # rotation and scale testing
# theta = 91
# scale = 1
# M = cv2.getRotationMatrix2D((cols/2, rows/2), theta, scale)
# dst = cv2.warpAffine(dst, M, (cols, rows))
#
# plt.subplot(321),plt.imshow(img, cmap='gray')
# plt.title('Input Image 1'), plt.xticks([]), plt.yticks([])
# plt.subplot(322),plt.imshow(dst, cmap='gray')
# plt.title('Input Image 2'), plt.xticks([]), plt.yticks([])
#
# ###########################
# # Fourier Transform
# dft_src = np.fft.fft2(img)
# dft_src_shift = np.fft.fftshift(dft_src)
#
# dft_dst = np.fft.fft2(dst)
# dft_dst_shift = np.fft.fftshift(dft_dst)
#
# ###########################
# # Compute maginitude and transfer to polar coordinate
# magnitude_spectrum_src = np.abs(dft_src_shift)
# magnitude_spectrum_dst = np.abs(dft_dst_shift)
#
# # plt.subplot(323),plt.imshow(np.log(magnitude_spectrum_src), cmap='gray')
# # plt.title('Magnitude Spectrum Image 1'), plt.xticks([]), plt.yticks([])
# # plt.subplot(324),plt.imshow(np.log(magnitude_spectrum_dst), cmap='gray')
# # plt.title('Magnitude Spectrum Image 2'), plt.xticks([]), plt.yticks([])
#
# # High pass filter
# magnitude_spectrum_src_hp = high_pass_filter(rows, cols) * magnitude_spectrum_src
# magnitude_spectrum_dst_hp = high_pass_filter(rows, cols) * magnitude_spectrum_dst
#
# plt.subplot(323),plt.imshow(magnitude_spectrum_src_hp, cmap='gray')
# plt.title('Magnitude Spectrum Image 1'), plt.xticks([]), plt.yticks([])
# plt.subplot(324),plt.imshow(magnitude_spectrum_dst_hp, cmap='gray')
# plt.title('Magnitude Spectrum Image 2'), plt.xticks([]), plt.yticks([])
#
# # change to log-polar
# polar_magnitude_spectrum_src = cv2.logPolar(magnitude_spectrum_src_hp, (rows/2, cols/2), 1, cv2.INTER_LINEAR)
# polar_magnitude_spectrum_dst = cv2.logPolar(magnitude_spectrum_dst_hp, (rows/2, cols/2), 1, cv2.INTER_LINEAR)
#
# ###########################
# # phase correlation
# scale0, theta0= phase_correlation(polar_magnitude_spectrum_src, polar_magnitude_spectrum_dst)
#
# # t_f_src = np.fft.fft2(polar_magnitude_spectrum_src)
# # t_f_dst = np.fft.fft2(polar_magnitude_spectrum_dst)
# #
# # a1 = np.angle(t_f_src)
# # a2 = np.angle(t_f_dst)
# #
# # theta_cross = np.exp(1j * (a1 - a2))
# # theta_phase = np.real(np.fft.ifft2(theta_cross))
# #
# # theta0, scale0 = np.unravel_index(theta_phase.argmax(), theta_phase.shape)
#
# DPP = 360 / rows
#
# theta0 = DPP * (theta0 - 1)
#
# scale0 = np.exp(scale0)
#
# print(theta0, scale0)
#
# plt.subplot(325),plt.imshow(polar_magnitude_spectrum_src, cmap='gray')
# plt.title('Log Polar Magnitude Spectrum Image 1'), plt.xticks([]), plt.yticks([])
# plt.subplot(326),plt.imshow(polar_magnitude_spectrum_dst, cmap='gray')
# plt.title('Log Polar Magnitude Spectrum Image 2'), plt.xticks([]), plt.yticks([])
# # plt.show()

#########################################################
# rotation and scale testing
theta = 45
scale = 0.5

M = cv2.getRotationMatrix2D((cols/2, rows/2), theta, scale)
dst = cv2.warpAffine(dst, M, (cols, rows))

plt.figure()
plt.subplot(321), plt.imshow(img, cmap='gray')
plt.title('Input Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(322), plt.imshow(dst, cmap='gray')
plt.title('Input Image 2'), plt.xticks([]), plt.yticks([])

###########################
# Fourier Transform
dft_src = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_src_shift = np.fft.fftshift(dft_src)

dft_dst = cv2.dft(np.float32(dst), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_dst_shift = np.fft.fftshift(dft_dst)

###########################
# Compute maginitude and transfer to polar coordinate
magnitude_spectrum_src = cv2.magnitude(dft_src_shift[:, :, 0], dft_src_shift[:, :, 1])
magnitude_spectrum_dst = cv2.magnitude(dft_dst_shift[:, :, 0], dft_dst_shift[:, :, 1])

plt.subplot(323), plt.imshow(np.log(magnitude_spectrum_src), cmap='gray')
plt.title('Magnitude Spectrum Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(324), plt.imshow(np.log(magnitude_spectrum_dst), cmap='gray')
plt.title('Magnitude Spectrum Image 2'), plt.xticks([]), plt.yticks([])

# High pass filter
magnitude_spectrum_src_hp = high_pass_filter(rows, cols) * magnitude_spectrum_src
magnitude_spectrum_dst_hp = high_pass_filter(rows, cols) * magnitude_spectrum_dst

plt.subplot(325), plt.imshow(magnitude_spectrum_src_hp, cmap='gray')
plt.title('Magnitude Spectrum Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(326), plt.imshow(magnitude_spectrum_dst_hp, cmap='gray')
plt.title('Magnitude Spectrum Image 2'), plt.xticks([]), plt.yticks([])

# change to log-polar
polar_src = cv2.logPolar(magnitude_spectrum_src_hp, (rows/2, cols/2), 1, cv2.INTER_LINEAR)
polar_dst = cv2.logPolar(magnitude_spectrum_dst_hp, (rows/2, cols/2), 1, cv2.INTER_LINEAR)

###########################
# phase correlation
# scale0, theta0 = phase_correlation(polar_src, polar_dst)

# phase correlation
theta_f1 = np.fft.fft2(polar_src)
theta_f2 = np.fft.fft2(polar_dst)


a1 = np.angle(theta_f1)
a2 = np.angle(theta_f2)

theta_cross = np.exp(1j * (a1 - a2))
theta_phase = np.real(np.fft.ifft2(theta_cross))

theta0, scale0 = np.unravel_index(theta_phase.argmax(), theta_phase.shape)

# DPP = 360 / cols
#
# theta0 = DPP * theta0
#
# scale0 = np.exp(scale0)

print(scale0, theta0)

# plt.figure()
# plt.subplot(121),plt.imshow(polar_src.T, cmap='gray')
# plt.title('Log Polar Magnitude Spectrum Image 1'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(polar_dst.T, cmap='gray')
# plt.title('Log Polar Magnitude Spectrum Image 2'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.imwrite('lena.png', img)
cv2.imwrite('lena_t.png', dst)

###########################################################################################

# img = sio.loadmat('pc_1.mat')
# img = img['L1']
# rows, cols = img.shape
#
# dst = sio.loadmat('pc_2.mat')
# dst = dst['L2']
#
# plt.figure()
# plt.subplot(121),plt.imshow(img, cmap='gray')
# plt.title('Log Polar Magnitude Spectrum Image 1'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(dst, cmap='gray')
# plt.title('Log Polar Magnitude Spectrum Image 2'), plt.xticks([]), plt.yticks([])
#
# # phase correlation
# theta_f1 = np.fft.fft2(img)
# theta_f2 = np.fft.fft2(dst)
#
# a1 = np.angle(theta_f1)
# a2 = np.angle(theta_f2)
#
# theta_cross = np.exp(1j * (a1 - a2))
# theta_phase = np.real(np.fft.ifft2(theta_cross))
#
# x0, y0 = np.unravel_index(theta_phase.argmax(), theta_phase.shape)
#
# ret = np.array([x0, y0])
# print(ret)
# print(np.round(ret))
# plt.show()
