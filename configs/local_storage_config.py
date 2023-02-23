import os

STORAGE = 'local'
# Definitions for COCO 2017 dataset
DATASET_PATH = "/datanfs/dataForOpenpose-YDL/dataset/"


IMAGES_PATH = "/datanfs/dataForOpenpose-YDL/dataset/conferenceroom/points2imgs"
TRAIN_ANNS = "/datanfs/dataForOpenpose-YDL/dataset/conferenceroom/train_annotations.json"
VALID_ANNS = "/datanfs/dataForOpenpose-YDL/dataset/conferenceroom/val_annotations.json"

# IMAGES_PATH = DATASET_PATH + "/train2017"
# TRAIN_ANNS = DATASET_PATH + "/annotations/person_keypoints_train2017.json"
# VALID_ANNS = DATASET_PATH + "/annotations\person_keypoints_val2017.json"



# will be used as output files
ROOT_TFRECORDS_PATH = "/datanfs/dataForOpenpose-YDL/dataset/TFrecords"
TRAIN_TFRECORDS = ROOT_TFRECORDS_PATH + "/training/"
VALID_TFRECORDS = ROOT_TFRECORDS_PATH + "/validation/"

RESULTS_ROOT ="/datanfs/dataForOpenpose-YDL/temp_weights/"
TENSORBOARD_PATH = RESULTS_ROOT + "/tensorboard/"
CHECKPOINTS_PATH = "/datanfs/dataForOpenpose-YDL/temp_weights/22Mon0822-2134/"
MODELS_PATH = "/datanfs/dataForOpenpose-YDL/trained_models/"
