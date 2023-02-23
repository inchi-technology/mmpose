import sys
import os
#tf.transpose
print("Python running from:",sys.prefix)
print("Current working dir",os.getcwd())

import sys
sys.path.append("/home/yq/YDL/openpose/")

import configs.default_config as cfg

import configs.local_storage_config as storage_cfg
#import remote_storage_config as storage_cfg
cfg.__dict__.update(storage_cfg.__dict__)

import tpu_training.TPU_config as TPU_config
#cfg.__dict__.update(TPU_config.__dict__) #comment out to disable TPU training  negative_log_likelyhood(x_real, x_real_mu))

cfg.RUN_NAME="ydl" #for reference


from utils import now
nowt=now()

import tensorflow as tf
print("version:",tf.__version__)
# configf = tf.ConfigProto()
# configf.gpu_options.allow_growth = True
# sess = tf.Session(config=configf)

#condaprint("Tensorflow version:",tf.version.VERSION)

# if cfg.TPU_MODE:
#     import tpu_training.init_TPU as init_TPU
#
#     print("Testing results bucket connectivity")
#     !touch / tmp / test
#     !gsutil
#     cp / tmp / test
#     {cfg.TENSORBOARD_PATH} / test
#     !gsutil
#     rm
#     {cfg.TENSORBOARD_PATH} / test
#     !gsutil
#     cp / tmp / test
#     {cfg.CHECKPOINTS_PATH} / test
#     !gsutil
#     rm
#     {cfg.CHECKPOINTS_PATH} / test
#     print("Testing dataset bucket connectivity")
#     !gsutil
#     ls
#     gs: // {cfg.GCS_TFRECORDS_BUCKETNAME} | head - 4
#     print("Testing TPU connectivity")
#     !nmap - Pn - p8470
#     {cfg.TPU_IP}
#     strategy, resolver = init_TPU.init_tpu(cfg.TPU_IP)  # This must be run before any imports!!!!



import datetime
import numpy as np

import dataset_functions
import models.six_stage_linear_model as model
import callbacks
import dataset_builder
import load_weights
import loss_metrics

# Training settings where
TRAINING_EPOCHS = 300
REAL_EPOCH_STEPS = int(cfg.DATASET_SIZE / cfg.BATCH_SIZE)
SHORT_EPOCH_STEPS=60 #actual epocsh used in training, smaller than real epoch, but allows to track progress better, [in batches]
SHORT_TRAINING_EPOCHS=int(TRAINING_EPOCHS*(REAL_EPOCH_STEPS/SHORT_EPOCH_STEPS))
SHORT_VALIDATION_STEPS=2 #per short epoch

EPOCH_RATIO=int(REAL_EPOCH_STEPS / SHORT_EPOCH_STEPS)
# adam_learning_rate=0.01 #for reference
BASE_LEARNING_RATE = 0.0005
LEARNING_RATE_SCHEDUELE = np.zeros(100000)  #used with short epochs
LEARNING_RATE_SCHEDUELE[:20] = 1
LEARNING_RATE_SCHEDUELE[20:300] = 0.7
LEARNING_RATE_SCHEDUELE[300 :500] = 0.5
LEARNING_RATE_SCHEDUELE[500 :700] = 0.3
LEARNING_RATE_SCHEDUELE[700:] = 0.1
LEARNING_RATE_SCHEDUELE *= BASE_LEARNING_RATE

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



if cfg.ASK_FOR_CHECKPOINTS:
    checkpoint,starting_epoch=load_weights.checkpoints_prompt(cfg)
else:
    checkpoint=None
    starting_epoch=0

model_maker = model.ModelMaker(cfg)  # must be outside scope to keep the graph clean
tf.keras.backend.clear_session()  # to clean to backaend from the imported model
from models.overall_model import overallModel




def define():
    train_model, test_model = model_maker.create_models()
    # overallmodel = overallModel(adaptor=adapt_model,generator=train_model)

    # if cfg.INCLUDE_MASK:
    #     losses=[loss_metrics.MaskedMeanAbsoluteError()
    #             ,loss_metrics.MaskedMeanAbsoluteError()
    #             ,loss_metrics.MaskedMeanAbsoluteError()
    #             ,loss_metrics.MaskedMeanAbsoluteError()
    #             ,loss_metrics.MaskedMeanSquaredError()
    #             ,loss_metrics.MaskedMeanSquaredError()]
    # else:
    #     raise NotImplementedError

    # this must match the model output order
    # tf.keras.metrics.categorical_crossentropy,
    metrics = [[tf.keras.metrics.categorical_crossentropy],
        [loss_metrics.MeanAbsoluteRatio(), loss_metrics.AnalogRecall()]
        , [loss_metrics.MeanAbsoluteRatio(), loss_metrics.AnalogRecall()]
        , [loss_metrics.MeanAbsoluteRatio(), loss_metrics.AnalogRecall()]
        , [loss_metrics.MeanAbsoluteRatio(), loss_metrics.AnalogRecall()]
        , [loss_metrics.MeanAbsoluteRatio(), loss_metrics.AnalogRecall()]
        , [loss_metrics.MeanAbsoluteRatio(), loss_metrics.AnalogRecall()]
    ]

    losses = [tf.keras.losses.mean_squared_error,
              loss_metrics.MaskedMeanSquaredError(),
              loss_metrics.MaskedMeanSquaredError(),
              loss_metrics.MaskedMeanSquaredError(),
              loss_metrics.MaskedMeanSquaredError(),
              loss_metrics.MaskedMeanSquaredError(),
              loss_metrics.MaskedMeanSquaredError()]

    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(BASE_LEARNING_RATE),
        #  loss=loss_metrics.MaskedMeanSquaredError()
        loss=losses
        , loss_weights=[0.1,5, 5, 5, 5, 1, 1]
        , metrics=metrics
    )
    return train_model, test_model


if cfg.TPU_MODE:
    print("TPU")
    with strategy.scope():
        train_model, test_model = define()
        if (checkpoint):
            train_model.load_weights(checkpoint)
else:
    print("gpu")
    train_model, test_model = define()
    if (checkpoint):
        train_model.load_weights(checkpoint)
        print("use checkpoint")

all_callbacks=[
    callbacks.make_LRscheduler_callback(LEARNING_RATE_SCHEDUELE)
    ,callbacks.PrintLR()
    ,tf.keras.callbacks.TerminateOnNaN()
]

if cfg.SAVE_CHECKPOINTS:
    checkpoint_callback,checkpoint_path=callbacks.make_checkpoint_callback(cfg,nowt,REAL_EPOCH_STEPS*cfg.BATCH_SIZE)
    all_callbacks.append(checkpoint_callback)
if cfg.SAVE_TENSORBOARD:
    tensorboard_callback,tensotboard_path=callbacks.make_tensorboard_callback(cfg,nowt,EPOCH_RATIO)
    all_callbacks+=[tensorboard_callback]
train_history=train_model.fit(
    dst
    ,epochs=1000
    ,steps_per_epoch=SHORT_EPOCH_STEPS
    ,validation_steps=SHORT_VALIDATION_STEPS
    ,validation_data=dsv
    ,callbacks=all_callbacks
    ,initial_epoch=0
)


tmp_path='E:/DeepLearning/Yet-Another-Openpose-Implementation-master/temp_weights/'+nowt+'/temp_weight-'+nowt
train_model.save_weights(tmp_path)
local_model_maker=model.ModelMaker(cfg) #must be outside scope to keep the graph clean
tf.keras.backend.clear_session() #to clean to backaend from the imported model

cpu_train_model,cpu_test_model=local_model_maker.create_models()

# cpu_train_model.load_weights(tmp_path)
cpu_test_model.load_weights(tmp_path)
cpu_test_model.save(cfg.MODELS_PATH+"\\test_"+cfg.RUN_NAME+nowt,include_optimizer=False)
print(cfg.RUN_NAME+nowt)
# cpu_train_model.save(cfg.MODELS_PATH+"\\train_"+cfg.RUN_NAME+nowt,include_optimizer=False)
