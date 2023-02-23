import json
import time


class labelprocessor:
    def __init__(self):
        self.id = 0;
        self.imgname =''
        self.kpttrans = \
            {
                "nose":"nose",
                "sternum":"",
                "Leye": "left_eye",
                "Reye":"right_eye",
                "Lear":"left_ear",
                "Rear":"right_ear",
                "Lshoulder":"left_shoulder",
                "Rshoulder": "right_shoulder",
                "Lelbow":"left_elbow",
                "Relbow": "right_elbow",
                "Lwrist": "left_wrist",
                "Rwrist":"right_wrist",
                "Lhip":"left_hip",
                "Rhip":"right_hip",
                "Lknee":"left_knee",
                "Rknee": "right_knee",
                "Lankle":"left_ankle",
                "Rankle": "right_ankle"
            }
        self.annotation = {"info":{},"licenses":[],"images":[],
             "categories":[
                 {
                    "skeleton": [
                        [16,14],
                        [14,12],
                        [17, 5],
                        [15,13],
                        [12,13],
                        [6 ,12],
                        [7 ,13],
                        [6 , 7],
                        [6 , 8],
                        [7 , 9],
                        [8 ,10],
                        [9 ,11],
                        [2 , 3],
                        [1 , 2],
                        [1 , 3],
                        [2 , 4],
                        [3 , 5],
                        [4 , 6],
                        [5 , 7]
                    ],
                    "name": "person", # 子类（具体类别）
                    "supercategory": "person", # 主类
                    "id": 1, # class id
                    "keypoints": [
                        "nose",
                        "left_eye",
                        "right_eye",
                        "left_ear",
                        "right_ear",
                        "left_shoulder",
                        "right_shoulder",
                        "left_elbow",
                        "right_elbow",
                        "left_wrist",
                        "right_wrist",
                        "left_hip",
                        "right_hip",
                        "left_knee",
                        "right_knee",
                        "left_ankle",
                        "right_ankle"
                    ]
            }],
             "annotations":[]}


    def write(self,kpts,imgshape,filename):
        if(not self.imgname == filename):
            self.imgname = filename
            self.annotation["images"].append(
                {
                    "id": int(filename.replace('.jpg','')),
                    "height": imgshape[0],
                    "width": imgshape[1],
                    "file_name": filename
                })
        self.id = self.id + 1
        self.annotation["annotations"].append(
            {
                "iscrowd": 0,
                "image_id":int(self.imgname.replace('.jpg','')),
                "num_keypoints":len(kpts),
                "id": self.id,
                "category_id": 1,
                "keypoints":[0]*17*3
            }

        )

        for kpt in kpts.keys():
            if kpt =="sternum":
                continue
            kpt_t = self.kpttrans[kpt]
            kpt_index = self.annotation["categories"][0]["keypoints"].index(kpt_t)
            self.annotation["annotations"][len(self.annotation["annotations"])-1]["keypoints"][3*kpt_index] = int (kpts[kpt][1]*imgshape[1])
            self.annotation["annotations"][len(self.annotation["annotations"])-1]["keypoints"][3 * kpt_index + 1] = int (kpts[kpt][0]*imgshape[0])
            self.annotation["annotations"][len(self.annotation["annotations"])-1]["keypoints"][3 * kpt_index + 2] = 2


    def save(self,name):
        f = open(name + "_annotations.json", "w")
        json.dump(self.annotation, f)
        f.close()

    def load(self,path_train,path_val):
        with open(path_train,'r') as f:
            result = json.load(f)
            self.annotation = result
        with open(path_val,'r') as f1:
            result = json.load(f1)
            for img in result['images']:
                self.annotation['images'].append(img)
            for ann in result['annotations']:
                self.annotation['annotations'].append(ann)
        self.anno = {}
        for anno in self.annotation['annotations']:
            imageid = str(anno['image_id'])
            if imageid in self.anno.keys():
                self.anno[imageid].append(anno['keypoints'])
            else:
                self.anno[imageid]=[anno['keypoints']]
        return self.anno