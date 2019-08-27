#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/22 10:31
# @Author   : Daishijun
# @Site     : 
# @File     : deepfm_Dense_single.py
# @Software : PyCharm


import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
# from deepctr.models import xDeepFM, DeepFM, DCN
from deepfm_Dense import DeepFM
from deepctr.inputs import SparseFeat, DenseFeat, get_fixlen_feature_names
# from keras.optimizers import Adam
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

data = pd.read_csv('./data/final_track2_train.txt', sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'device', 'time', 'duration_time'])
sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'device']
dense_features = ['time', 'duration_time']

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0,)

target = ['finish']
# target = ['like']

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

## 这里有多余的步骤，该方法中间为每个特征设置了Input层，但是没有返回，只返回了特征名称list，其实可以直接从上面的两个list合并得到。
feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)


train, test = train_test_split(data, test_size=0.1)
train_model_input = [train[name] for name in feature_names]
test_model_input = [test[name] for name in feature_names]

#model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary',use_fm=True,
               dnn_hidden_units=(128,128), dnn_dropout=0)
# model = DCN(dnn_feature_columns, embedding_size=8)
model.compile(Adam(lr=0.005), "binary_crossentropy", metrics=['binary_crossentropy'], )
#es = EarlyStopping(monitor='val_binary_crossentropy')
history = model.fit(train_model_input, train[target].values, validation_split=0.3, callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')],
                    batch_size=4096, epochs=10, verbose=1)

pred_ans = model.predict(test_model_input, batch_size=2**14)
pred_finish = (pred_ans*2).astype(int)
print("test accuracy", round(accuracy_score(test[target].values, pred_finish), 4))
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))