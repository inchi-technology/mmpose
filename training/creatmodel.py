import tensorflow as tf
import models.six_stage_linear_model as model
import sys
sys.path.append("E:/DeepLearning/Yet-Another-Openpose-Implementation-master/configs/")
import default_config as cfg
import local_storage_config as storage_cfg
cfg.__dict__.update(storage_cfg.__dict__)
from utils import now
import numpy as np

import dataset_functions
import models.six_stage_linear_model as model
import callbacks
import dataset_builder
import load_weights
import loss_metrics
nowt=now()
model_ds=model.ModelDatasetComponent(cfg)
tfrecord_files_train=dataset_builder.get_tfrecord_filenames(cfg.TRAIN_TFRECORDS,cfg)
tfrecord_files_valid=dataset_builder.get_tfrecord_filenames(cfg.VALID_TFRECORDS,cfg)
print("Found the following training TFrecords:\n","\n".join(tfrecord_files_train))
print("Found the following validation TFrecords:\n","\n".join(tfrecord_files_valid))
print("Building training dataset")
dst=dataset_builder.build_training_ds(tfrecord_files_train,model_ds.place_training_labels,cfg)
print("Training dataset shape:",dst)
print("Building validation dataset")
dsv=dataset_builder.build_validation_ds(tfrecord_files_valid,model_ds.place_training_labels,cfg)
print("Validation dataset shape:",dsv)

metrics = [
    [loss_metrics.MeanAbsoluteRatio(), loss_metrics.AnalogRecall()]
    , [loss_metrics.MeanAbsoluteRatio(), loss_metrics.AnalogRecall()]
    , [loss_metrics.MeanAbsoluteRatio(), loss_metrics.AnalogRecall()]
    , [loss_metrics.MeanAbsoluteRatio(), loss_metrics.AnalogRecall()]
    , [loss_metrics.MeanAbsoluteRatio(), loss_metrics.AnalogRecall()]
    , [loss_metrics.MeanAbsoluteRatio(), loss_metrics.AnalogRecall()]
]

all_callbacks=[
    callbacks.PrintLR()
    ,tf.keras.callbacks.TerminateOnNaN()
]
tmp_path='E:/DeepLearning/Yet-Another-Openpose-Implementation-master/temp_weights/temp_weight.pb'
train_model=tf.keras.models.load_model('E:\\DeepLearning\\Yet-Another-Openpose-Implementation-master\\trained_models\\test_ydl22Tue0222-1150')
train_model.compile(optimizer=tf.keras.optimizers.Adam(0.001)
                        , loss = loss_metrics.MaskedMeanSquaredError()


)
train_history=train_model.fit(
    dst
    ,epochs=2
    ,steps_per_epoch=50
    ,validation_steps=5
    ,validation_data=dsv
    ,callbacks=all_callbacks
    ,initial_epoch=0
)
train_model.save(cfg.MODELS_PATH+"\\train_continue-"+cfg.RUN_NAME+nowt,include_optimizer=False)