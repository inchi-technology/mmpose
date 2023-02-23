import json
# 创建字典
import os
import numpy as np
# JSON到字典转化
# playground-road  90  91
# playground-indoor 152   57
# road-office 186  23
path = './dataset/laboratory/'
files = os.listdir(path)
filenames = [x for x in files if 'json' in x]

for filename in filenames:
    print(filename)
    filepath = path +filename
    with  open(filepath, 'r') as f2:
        info_data = json.load(f2)
        data = {}
        means = []
        for k in info_data.keys():
            if k in ['Lear','Rear','Leye','Reye']:
                continue
            data[k] = [np.mean( info_data[k][0]),np.mean( info_data[k][1]),np.mean( info_data[k][2])]
            means.append(data[k])
    # 显示数据类型
        print(data)
        print(np.mean(axis=0,a=means))

