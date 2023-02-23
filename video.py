import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import random
import sys


def base_path(path):
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    elif __file__:
        application_path = os.path.dirname(__file__)
    return os.path.join(application_path, path)


basepath = base_path('')
print(basepath)
import argparse
import cv2
import visualizations as vis
from applications.model_wrapper import ModelWrapper
from writelabel import labelprocessor
import configs.draw_config as draw_config
import numpy as np

model_path = "./trained_models/test_ydl21Tue0223-1625(laboratory)"


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # 备注root返回当前目录路径；dirs返回当前路径下所有子目录；files返回当前路径下所有非目录子文件
        return files


class VideoApp:
    def __init__(self, input_video_file, output_filename, fourcc_str, fps):
        assert len(fourcc_str) == 4

        self.model_wrapper = ModelWrapper(model_path)
        self.video_reader = cv2.VideoCapture(input_video_file)
        # if not self.video_reader.isOpened():
        #     raise IOError("Error opening video file")
        # height, width = self.get_video_size()
        self.fps = fps
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.video_writer = cv2.VideoWriter(output_filename, cv2.CAP_FFMPEG, fourcc, fps, (640, 480))
        self.labelwriter = labelprocessor()  # 初始化标签
        self.labelloader = labelprocessor()
        # self.labelloader.load('./dataset/laboratory/val_annotations.json','./dataset/laboratory/train_annotations.json')
        self.msecaulator = MSE(self.labelloader.kpttrans.keys())

    def get_video_size(self):
        ret, img_bgr = self.video_reader.read()
        return img_bgr.shape[0], img_bgr.shape[1]

    def showmidResault(self, pafs, kpts):
        pafid_x = 10
        pafid_y = 27

        Lshoulder_id = 10
        Lelbow_id = 11

        Lshoulder_map = kpts[..., Lshoulder_id]
        Lelbow_map = kpts[..., Lelbow_id]
        Lshoulder_map = np.power(Lshoulder_map, 18)
        Lshoulder_map = cv2.resize(Lshoulder_map, (640, 480))
        Lshoulder_map = (Lshoulder_map * 255).astype(np.uint8)
        # a = np.where(LShoulder_map>240,LShoulder_map,0)
        # cv2.imshow("map",Lshoulder_map)
        # cv2.waitKey(1)
        return Lshoulder_map

    def process_frame(self, img, filename):
        skeletons, pafs, kpts = self.model_wrapper.process_image(img)
        skeleton_drawer = vis.SkeletonDrawer(img, draw_config)
        try:
            self.caculate_dis(skeletons, filename.replace('.jpg', ''))
        except Exception as e:
            print(e)
        for skeleton in skeletons:
            self.labelwriter.write(kpts=skeleton.keypoints, imgshape=img.shape, filename=filename)
            skeleton.draw_skeleton(skeleton_drawer.joint_draw, skeleton_drawer.kpt_draw)
        return img, pafs, kpts

    def caculate_dis(self, skeletons, imageid):
        self.msecaulator.cacuSingleDistance(imageid, skeletons, self.labelloader.anno)

    def run(self, skip):
        output_path = './analysis/road-office/origin/'

        if (not os.path.exists(output_path)):
            os.makedirs(output_path)
        imgoutput_path = output_path + 'road/'

        if (not os.path.exists(imgoutput_path)):
            os.makedirs(imgoutput_path)

        depthpath = ['/datanfs/dataForOpenpose-YDL/dataset/conferenceroom/points2imgs/']

        # depthpath = "./mmw_data_sky/saveimages/"
        imgpath = ['/datanfs/dataForOpenpose-YDL/dataset/conferenceroom/saveimages/']
        files = []
        for path in depthpath:
            cfile = os.listdir(path)
            files = files + cfile

        print("Processing video")
        print("Press ESC to exit\n")
        cv2.namedWindow("video-process", cv2.WINDOW_AUTOSIZE)
        count = 0
        thred = int(0.6 * len(files))
        i = 0
        # while True:
        # np.random.seed(10)
        # np.random.shuffle(files)
        print(len(files))
        for file in files:
            # ret, img_bgr = self.video_reader.read()
            # if skip and skip > 0:
            #     skip -= 1
            #     continue
            # if not ret:
            #     break
            # file = '166270995727.jpg'
            img_bgr = cv2.imread([path + file for path in depthpath if os.path.exists(path + file)][0])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            processed_img_rgb, pafs, kpts = self.process_frame(img_rgb, file)
            # Lshouder_map = self.showmidResault(pafs,kpts)
            processed_img_bgr = cv2.cvtColor(processed_img_rgb, cv2.COLOR_RGB2BGR)

            # self.video_writer.write(processed_img_bgr)
            img = cv2.imread([path + file for path in imgpath if os.path.exists(path + file)][0])
            print(file)
            mse = {}
            for k in self.msecaulator.mse.keys():
                if len(self.msecaulator.mse[k]) == 3:
                    mse[k] = [round(np.mean(self.msecaulator.mse[k][0]), 2),
                              round(np.mean(self.msecaulator.mse[k][1]), 2),
                              round(np.mean(self.msecaulator.mse[k][2]), 2)]
                else:
                    mse[k] = [-1, -1, -1]
            print(mse)
            # map = np.zeros(shape=(480,640,3))
            # # map[...,0] = Lshouder_map
            # # map[...,1] = Lshouder_map
            # map[...,2] = Lshouder_map
            # map = map.astype(np.uint8)
            # cv2.imshow("video-process", cv2.add(map,cv2.add(processed_img_bgr,img)))
            cv2.imshow("video-process", cv2.add(processed_img_bgr, img))

            # cv2.imwrite(imgoutput_path+file,cv2.add(processed_img_bgr,img))

            key = cv2.waitKey(1)
            if key == 27:  # Esc key to stop
                break
            print(".", end="", flush=True)
            if not i % self.fps:
                print("-", i / self.fps, count, thred, len(files))
            if count == thred:
                # self.labelwriter.save("train")
                self.labelwriter = None
                self.labelwriter = labelprocessor()
            count = count + 1
        # self.labelwriter.save("val")
        cv2.destroyWindow("video-process")
        jsonpath = output_path + 'json/ '
        if not os.path.exists(jsonpath):
            try:
                os.makedirs(jsonpath)
            except Exception as e:
                print(e)

        # self.msecaulator.save(jsonpath+'road-origin.json')
        self.video_reader.release()
        self.video_writer.release()


class MSE:
    def __init__(self, partlist):
        self.mse = {}
        self.partlist = list(partlist)
        for part in self.partlist:
            self.mse[part] = [[], [], []]
        self.partlist.remove("sternum")

    def cacuSingleDistance(self, imgid, preds, annos):
        for pred in preds:
            for kpt in pred.keypoints.keys():
                mse = -1
                cur_x = 0
                cur_y = 0
                for anno in annos[imgid]:

                    if kpt == 'sternum':
                        if anno[3 * self.partlist.index('Lshoulder')] == 0 and anno[
                            3 * self.partlist.index('Rshoulder')] == 0:
                            continue
                        x_true = anno[3 * self.partlist.index('Lshoulder')] / 2 + anno[
                            3 * self.partlist.index('Rshoulder')] / 2
                        y_true = anno[3 * self.partlist.index('Lshoulder') + 1] / 2 + anno[
                            3 * self.partlist.index('Rshoulder') + 1] / 2
                    else:
                        index = self.partlist.index(kpt)
                        x_true = anno[3 * index]
                        y_true = anno[3 * index + 1]
                    x_pred = pred.keypoints[kpt][1] * 640
                    y_pred = pred.keypoints[kpt][0] * 480

                    mse_x = (x_pred - x_true) ** 2
                    mse_y = (y_pred - y_true) ** 2
                    curmse = (mse_x + mse_y) ** 0.5
                    if ((mse == -1) or mse > curmse):
                        cur_x = x_pred - x_true
                        cur_y = y_pred - y_true
                        mse = curmse
                self.mse[kpt][0].append(int(cur_x))
                self.mse[kpt][1].append(int(cur_y))
                self.mse[kpt][2].append(int(mse))

    def save(self, path):
        import json
        # 创建字典
        # dumps 将数据转换成字符串
        if os.path.exists(path):
            os.remove(path)
        info_json = json.dumps(self.mse)
        f = open(path, 'w')
        f.write(info_json)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video with "Yet another Openpose implementation".')
    parser.add_argument('--input', type=str, required=False, default='350243419-1-64.flv', help='The video to process')
    parser.add_argument('--output', type=str, required=False, default='./pointcloud.mp4', help='The output filename')
    parser.add_argument('--fourcc', type=str, required=False, default='mp4v',
                        help='optional fourcc codec code (must be installed and usable by OpenCV)')
    parser.add_argument('--fps', type=float, required=False, default=30, help='optional input video fps')
    parser.add_argument('--skip', type=int, required=False, help='optional number of frames to skip')
    args = parser.parse_args()

    app = VideoApp(args.input, args.output, args.fourcc, args.fps)
    app.run(args.skip)
