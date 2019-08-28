import os
import time
import numpy as np
import cv2

from dataloader.dataloader import MyDataloader

if __name__ == '__main__':
    print("hello world!")

    root = '/home/kuanghf/Seafile/dataset/simulation/yaw'
    data = MyDataloader(root)

    # unwrap raw images
    start_time = time.time()
    data.unwrap_dataset()
    print("--- %s seconds ---" % (time.time() - start_time))

    # checking unwrap results
    for i in range(len(data.unwrap_images)):
        cv2.imshow('img', data.unwrap_images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

