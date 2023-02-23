import json
import random

import scipy.io as scipyio
import cv2 as cv
import os
import sys
import numpy as np
import yaml
from matplotlib import pyplot as plt
import scipy.io as scipyio
import cv2 as cv
import os
import sys
from matplotlib import animation
import numpy as np
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        #备注root返回当前目录路径；dirs返回当前路径下所有子目录；files返回当前路径下所有非目录子文件
        return files
filepath = "./mmw_data_sky/points/"
imgpath = "./mmw_data_sky/saveimages/"
files = file_name(filepath)
kx = 455.17505077311426
ky = -456.90585642530476
u0 = 319.5533277238348
v0 = 240.38912331730117
fig = plt.figure(figsize=(20, 10))

for filename in files[1:]:
        # if(filename <= "165993984567.mat"):
        #     continue
        path = filepath + filename
        mat = scipyio.loadmat(path)['points']
        mat[:, 0] = (mat[:, 0] / mat[:, 1]) * kx + u0+74
        mat[:, 2] = (mat[:, 2] / mat[:, 1]) * ky + v0+21
        mat[:, 1] = mat[:, 1] * 255 / 6
        mat = mat.astype(np.int32)
        picmat = np.zeros((480,640,3))
        pointnum = 0
        for point in mat:
            if  point[2]<480 and point[0]<640 and point[2]>=0 and point[0]>=0 and picmat[point[2], point[0], 0] ==0 :
                picmat[point[2], point[0], 0] = point[1]
                picmat[point[2], point[0], 1] = point[1]
                picmat[point[2], point[0], 2] = point[1]
                pointnum+=1
        # filename = filename.replace(".mat", ".jpg")
        # cv.imwrite(imgpath+filename, picmat)
        # print(filename+" saved!")

        print(str(mat.shape[0]) + "-->" + str(pointnum))
        picmat = picmat.astype(np.uint8)
        singleimgpath = imgpath+filename.replace(".mat",".jpg")
        img = cv.imread(singleimgpath)
        img = cv.add(img,picmat)
        res = cv.addWeighted(img, 0.7, picmat, 0.3, 0)
        fig.canvas.set_window_title(filename)
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(picmat)
        plt.pause(0.01)
        plt.clf()
        # cv.namedWindow('Kawaii Small Animals', cv.WINDOW_NORMAL)
        # cv.imshow('Kawaii Small Animals',img)
        # cv.waitKey(50)
plt.show()

