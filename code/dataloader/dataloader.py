import cv2
import numpy as np
import time


# omni-lens with android_ros app param
Cx = 695
Cy = 350
SR = 120
LR = 230

imagelist = ['imageLists.txt', 'imageLists_step_2.txt', 'imageLists_step_5.txt', 'imageLists_step_10.txt']


# for loading raw datasets and panorama it to a general images
class MyDataloader:
    def __init__(self, root):
        self.path = root
        self.raw_images = []
        self.unwrap_images = []

        # reading lines for extract images
        file = self.path + '/' + imagelist[0]
        with open(file, 'r') as f:
            for line in f.readlines():
                image_path = self.path + '/' + line.strip('\n')
                # print(image_path)

                img_temp = cv2.imread(image_path, 0)
                self.raw_images.append(img_temp)

        # unwrape image parameters
        self.map_x, self.map_y = None, None

    def set_unwrap_param(self, img):
        Hd = int(LR - SR)
        Wd = int(np.pi * (SR + LR) + 1)

        dst = np.zeros([Hd, Wd])
        map_x = np.zeros(dst.shape, np.float32)
        map_y = np.zeros(dst.shape, np.float32)

        start_time = time.time()
        for j in range(dst.shape[0]):
            for i in range(dst.shape[1]):
                r= j / Hd *(LR - SR) + SR
                theta = i / Wd * 2 * np.pi
                Xs = Cx + r * np.sin(theta)
                Ys = Cy + r * np.cos(theta)
                map_x[j, i] = Xs
                map_y[j, i] = Ys
        print("--- %s seconds ---" % (time.time() - start_time))

        self.map_x, self.map_y = map_x, map_y

    def unwrap_dataset(self):
        self.set_unwrap_param(self.raw_images[0])

        for i in range(len(self.raw_images)):
            # img_temp = self.unwrap_image(self.raw_images[i])
            img_temp = cv2.remap(self.raw_images[i], self.map_x, self.map_y, cv2.INTER_LINEAR)

            self.unwrap_images.append(img_temp)



