#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/22 16:53
# @Author   : Daishijun
# @Site     : 
# @File     : deepfmMMoetrain.py
# @Software : PyCharm



import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from deepctr.inputs import SparseFeat, DenseFeat, get_fixlen_feature_names
# from keras.optimizers import Adam
from tensorflow.python.keras.optimizers import Adam
# from tensorfl import Adam
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils import multi_gpu_model
from deepfm_add_MMoe import DeepFM
# from train2modelmmoe import DeepFMmmoe
from tensorflow.python.keras.utils import plot_model


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

DATA_PATH = '/opt/ByteCamp/'
DATA_FILE = 'bytecamp.data'

CHECKPOINT_ROOT_DIR = './checkpoint/camp'

data = pd.read_csv(DATA_PATH+DATA_FILE, sep=',')
sparse_features = ['uid', 'u_region_id', 'item_id', 'author_id','music_id']
dense_features = ['duration', 'generate_time']

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0,)

target = ['finish', 'like']
# target = ['finish']

data['generate_time'] %= 60 * 60 * 24



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


RIGIONID = 0
# train_indexs = data[(data['date'] < 20190708) & (data['u_region_id']==RIGIONID)].indexd



# test_indexs = data[(data['date'] == 20190708) & (data['u_region_id']==RIGIONID)].index


# train, test = data.loc[train_indexs], data.loc[test_indexs]
#
# train_model_input = [train[name] for name in feature_names]
#
# test_model_input = [test[name] for name in feature_names]

train_indexs = data[data['date'] < 20190707].index

test_indexs = data[data['date'] == 20190708].index
valid_indexs = data[data['date'] == 20190707].index

train, test = data.loc[train_indexs], data.loc[test_indexs]
valid = data.loc[valid_indexs]

train_model_input = [train[name] for name in feature_names]
test_model_input = [test[name] for name in feature_names]
valid_model_input = [valid[name] for name in feature_names]



from tensorflow.python.keras import backend as K

# def auc(y_true, y_pred):
#     auc = tf.metrics.auc(y_true, y_pred)[1]
#     K.get_session().run(tf.local_variables_initializer())
#     return auc





model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary',use_fm=True,
               dnn_hidden_units=(128,256), dnn_dropout=0)

# model = DeepFMmmoe(linear_feature_columns, dnn_feature_columns, embedding_size=8, use_fm=True, dnn_hidden_units=(128, 128,),
#            l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0,
#            dnn_activation='relu', dnn_use_bn=False, task='binary')

# try:
#     model = multi_gpu_model(model, gpus=2)
#     print("Training using multiple GPUs..")
# except Exception as e:
#     print(e)
#     print("Training using single GPU or CPU..")

model.compile(Adam(lr=0.0001), "binary_crossentropy", metrics=['binary_crossentropy', tf.python.keras.metrics.AUC()], loss_weights=[0.6,0.4])

model.summary()

print(model.metrics_names)

checkpoint_path = os.path.join(CHECKPOINT_ROOT_DIR, "deepfmMMoetrain", "deepfmMMoetrain-{epoch:04d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, monitor='val_finish_auc',
                                       save_weights_only=True, save_best_only=True, mode='max'),
    tf.keras.callbacks.EarlyStopping(monitor='val_finish_auc', patience=2, verbose=1)
]

plot_model(model, to_file='deepfmMMoEmodel.png', show_shapes=True, show_layer_names=True)

history = model.fit(train_model_input, {"finish":train["finish"].values, "like":train["like"].values},
                    batch_size=4096, epochs=10, verbose=1, validation_data=(valid_model_input,
                                     {"finish": valid["finish"].values, "like": valid["like"].values}),
                    callbacks =callbacks)

latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest is not None:
    model.load_weights(latest)
    print("load model from %s" % (latest))


pred_ans_finish, pred_ans_like = model.predict(test_model_input, batch_size=2**14)
pred_finish = (pred_ans_finish*2).astype(int)
pred_like = (pred_ans_like*2).astype(int)


print('Region ID:', RIGIONID)
print('test accuracy ==>  finish:{} \t like:{}'.format(round(accuracy_score(test['finish'].values, pred_finish), 4),\
                                                      round(accuracy_score(test['like'].values, pred_like), 4)))
print('test LogLoss ==>  finish:{} \t like:{}'.format(round(log_loss(test['finish'].values, pred_ans_finish), 4),\
                                                      round(log_loss(test['like'].values, pred_ans_like), 4)))
print('test AUC ==>  finish:{} \t like:{}'.format(round(roc_auc_score(test['finish'].values, pred_ans_finish), 4),\
                                                     round(roc_auc_score(test['like'].values, pred_ans_like), 4)))