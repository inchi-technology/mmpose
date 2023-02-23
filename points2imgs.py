import scipy.io as scipyio
import cv2 as cv
import os
import sys
import numpy as np
# 将点二维化
filepath = "./mmw_data_sky/points/"
imgpath = "./mmw_data_sky/saveimages/"
point2imgpath = "./mmw_data_sky/points2imgs/"
kx = 455.17505077311426
ky = -456.90585642530476
u0 = 319.5533277238348
v0 = 240.38912331730117

def process():
    sum = 0
    files = os.listdir(filepath)
    for filename in files:
            path = filepath + filename
            mat = scipyio.loadmat(path)["points"]
            mat[:, 0] = (mat[:, 0] / mat[:, 1]) * kx + u0 + 74
            mat[:, 2] = (mat[:, 2] / mat[:, 1]) * ky + v0 + 21
            mat[:, 1] = mat[:, 1] * 255 / 6
            mat = mat.astype(np.int32)
            picmat = np.zeros((480,640,3))
            num = 0
            for point in mat:
                if point[2]<480 and point[0]<640 and point[2]>=0 and point[0]>=0:
                    picmat[point[2], point[0], 0] = point[1]
                    picmat[point[2], point[0], 1] = point[1]
                    picmat[point[2], point[0], 2] = point[1]
                    num += 1
            picmat = picmat.astype(np.uint8)
            filename = filename.replace(".mat", ".jpg")
            if(num>40):
                cv.imwrite(point2imgpath+filename, picmat)
                print(filename+" saved!"+   "    " + str(len(mat)) + "    " + str(num))
                sum += 1
            # 下面删除点数不足40的文件，怕误删这行代码可以注释掉
            else :
                os.remove(imgpath + filename )
                os.remove(filepath + filename.replace(".jpg", ".mat"))
    print("一共" + str(sum) + "帧点云转换为图片")


def processOneFrame(mat):
    mat[:, 0] = (-mat[:, 0] / mat[:, 1] + 1) * kx + u0
    mat[:, 2] = (mat[:, 2] / mat[:, 1] + 1) * ky + v0
    mat[:, 1] = mat[:, 1] * 255 / 6
    mat = mat.astype(np.int32)
    picmat = np.zeros((480, 640, 3))
    num = 0
    for point in mat:
        if point[2] < 480 and point[0] < 640 and point[2] >= 0 and point[0] >= 0:
            picmat[point[2], point[0], 0] = point[1]
            picmat[point[2], point[0], 1] = point[1]
            picmat[point[2], point[0], 2] = point[1]
            num += 1
    picmat = picmat.astype(np.uint8)
    return picmat

process()

