import cv2
import numpy as np


# generating a high pass filter mask
def high_pass_filter(ht, wd):
    eta = np.cos(np.pi * np.linspace(-0.5, 0.5, num=ht)).reshape(1, ht)
    neta = np.cos(np.pi * np.linspace(-0.5, 0.5, num=wd)).reshape(1, wd)

    X = np.dot(eta.T, neta)

    H = (1.0 - X) * (2.0 - X)

    return H


def log_polar_param(img):
    Y, X = img.shape

    # find center
    Xc = (1 + X) / 2
    Yc = (1 + Y) / 2

    # Calculate log base and angle increments
    base = np.exp(np.log(Xc - 1) / (Xc - 1))
    dtheta = np.pi / Y

    # Build x-y coordinates of log-polar points
    Xin = np.zeros([Y, X], np.float32)
    Yin = np.zeros([Y, X], np.float32)

    for y in range(Y):
        theta = y * dtheta
        for x in range(X):
            r = base ** (x / 2) - 1
            Xin[y, x] = r * np.cos(theta) + Xc
            Yin[y, x] = r * np.sin(theta) + Yc

    return Xin, Yin, base


# Converting image coordinate to log-polar coordinate
def log_polar(img, Xin, Yin, base):
    img_lp = None

    # remap
    img_lp = cv2.remap(img, Xin, Yin, cv2.INTER_CUBIC)

    img_lp = np.nan_to_num(img_lp)

    return img_lp, base
