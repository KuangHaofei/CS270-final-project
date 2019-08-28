import numpy as np
import cv2


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


img_r = cv2.imread('lena.png', 0)
rows, cols = img_r.shape

tx = 105.
ty = -115.

M_T = np.float32([[1, 0, tx],
                  [0, 1, ty]])

img_s = cv2.warpAffine(img_r, M_T, (cols, rows))

x0, y0 = estimation_translation(img_r, img_s)

print(x0, y0)