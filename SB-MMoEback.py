#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/22 19:23
# @Author   : Daishijun
# @Site     : 
# @File     : SB-MMoEback.py
# @Software : PyCharm

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from deepctr.models import xDeepFM, DeepFM, DCN
from deepctr.inputs import SparseFeat, DenseFeat, get_fixlen_feature_names
from tensorflow.python.keras.optimizers import Adam

from deepctr.inputs import input_from_feature_columns, get_linear_logit,build_input_features,combined_dnn_input
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import FM
from deepctr.layers.utils import concat_fun
from mmoe import MMoE
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.python.keras.layers import   Input
from tensorflow.python.keras.utils import multi_gpu_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
config = tf.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.Session(config=config)

data = pd.read_csv('./data/final_track2_train.txt', sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'device', 'time', 'duration_time'])
sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'device']
dense_features = ['time', 'duration_time']

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0,)

# target = ['finish', 'like']
target = ['finish', 'like']

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

sparse_feature_columns = [SparseFeat(feat, data[feat].nunique())  #(特征名, 特征不同取值个数)生成SparseFeat对象，name == 特征名，dimension==该特征不同取值个数， dtype ==int32
                        for feat in sparse_features]
dense_feature_columns = [DenseFeat(feat, 1)  #（特征名， dimension==1） 数据dtype == float32
                      for feat in dense_features]
dnn_feature_columns = sparse_feature_columns + dense_feature_columns
linear_feature_columns = sparse_feature_columns + dense_feature_columns

##['feature1','feature2',...]
feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)


train, test = train_test_split(data, test_size=0.1)


train_model_input = [train[name] for name in feature_names]
test_model_input = [test[name] for name in feature_names]


features = build_input_features(linear_feature_columns + dnn_feature_columns)

inputs_list = list(features.values())
sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                     embedding_size=8,
                                                                     l2_reg=0.00001, init_std=0.0001,
                                                                     seed=1024)

dnn_input = combined_dnn_input(sparse_embedding_list,dense_value_list)
# print('test_model_input info')
# print(len(test_model_input))
# print(type(test_model_input[0]))
# print(len(test_model_input[0]))
# print('num_features:',len(test_model_input[0]) )


# MMoE
mmoe_input = tf.keras.layers.Dense(units=128,activation='relu')(dnn_input)
mmoe_layers = MMoE(units=16, num_experts=8, num_tasks=2)(mmoe_input)

print('passed')
output_layers = []

# Build tower layer from MMoE layer
output_info = ['finish', 'like']
for index, task_layer in enumerate(mmoe_layers):
    tower_layer = tf.keras.layers.Dense(
        units=128,
        activation='relu'
        )(task_layer)
    output_layer = tf.keras.layers.Dense(
        units=1,
        name=output_info[index],
        activation='sigmoid'
        )(tower_layer)

    output_layers.append(output_layer)

model = tf.keras.Model(inputs=inputs_list, outputs=output_layers)


try:
    model = multi_gpu_model(model, gpus=2)
    print("Training using multiple GPUs..")
except Exception as e:
    print(e)
    print("Training using single GPU or CPU..")

model.compile(optimizer=Adam(0.0001), loss={'finish': 'binary_crossentropy', 'like':'binary_crossentropy'},\
              metrics=['accuracy','binary_crossentropy'], loss_weights={'finish':0.6, 'like':0.4})

model.summary()


# class_wights = [{0:0.5, 1:0.5}, {0:0.6,1:0.4}]
history = model.fit(x=train_model_input, y={'finish':train['finish'].values, 'like':train['like'].values},
                    validation_split=0.3,callbacks=[EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')],\
                    batch_size=4096, epochs=20, verbose=1)

pred_ans_finish, pred_ans_like = model.predict(test_model_input, batch_size=2**14)
pred_finish = (pred_ans_finish*2).astype(int)
pred_like = (pred_ans_like*2).astype(int)
print('test accuracy ==>  finish:{} \t like:{}'.format(round(accuracy_score(test['finish'].values, pred_finish), 4),\
                                                      round(accuracy_score(test['like'].values, pred_like), 4)))
print('test LogLoss ==>  finish:{} \t like:{}'.format(round(log_loss(test['finish'].values, pred_ans_finish), 4),\
                                                      round(log_loss(test['like'].values, pred_ans_like), 4)))
print('test AUC ==>  finish:{} \t like:{}'.format(round(roc_auc_score(test['finish'].values, pred_ans_finish), 4),\
                                                      round(roc_auc_score(test['like'].values, pred_ans_like), 4)))