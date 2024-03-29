import cv2
import numpy as np
import utils


# iFMI Visual Odometry Class
# @param img_r: input image, a 2D array of numpy uint8 type;
# @parma img_s: input image, a 2D array of numpy uint8 type;
# s is generated by r through geometry transformation(translation, rotation, scaling)
class VOiFMI:
    def __init__(self, img_r=None, img_s=None):
        # two registration images
        self.img_r = img_r
        self.img_s = img_s

        self.rows, self.cols = self.img_r.shape

        # registration parameters
        self.tx = 0
        self.ty = 0
        self.theta = 0
        self.scale = 1

        # pose : x, y, z, raw, pith, yaw
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.raw = 0.
        self.pitch = 0.
        self.yaw = 0.

        self.pose = np.zeros(6)

        # path
        self.path = []

    def print_state(self):
        # print registration parameters
        print('Registration paramters are [tx, ty, theta, scale]: [',
              self.tx, self.ty, self.theta, self.scale, ']')

        # print pose state
        print('pose estimation[x, y, z, raw, pith, yaw] : [',
              self.x, self.y, self.raw, self.pitch, self.yaw, ']')

    def set_img(self, img_r, img_s):
        self.img_r = img_r
        self.img_s = img_s

    # for computing
    def estimation_translation(self):
        rows, cols = self.img_r.shape
        img_r = cv2.copyMakeBorder(self.img_r, rows, rows, cols, cols, cv2.BORDER_CONSTANT, value=0)
        img_s = cv2.copyMakeBorder(self.img_s, rows, rows, cols, cols, cv2.BORDER_CONSTANT, value=0)

        img_r = np.float32(img_r)
        img_s = np.float32(img_s)

        [x0, y0], _ = cv2.phaseCorrelate(img_r, img_s)

        if abs(x0) > rows or abs(y0) > cols:
            x0, y0 = np.nan, np.nan

        self.tx = x0
        self.ty = y0

    def estimation_rotation_scaling(self):
        rows, cols = self.img_r.shape
        # fourier transfer
        f_r = np.fft.fft2(self.img_r)
        f_r_shift = np.fft.fftshift(f_r)
        f_r_mag = np.abs(f_r_shift)

        f_s = np.fft.fft2(self.img_s)
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
        lp_r, _ = utils.log_polar(f_r_hp)
        cp_r = np.fft.fft2(lp_r)

        lp_s, base = utils.log_polar(f_s_hp)
        cp_s = np.fft.fft2(lp_s)

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

        self.theta = theta
        self.scale = scaling

        return theta, scaling
