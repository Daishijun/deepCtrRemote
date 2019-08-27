#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/13 13:02
# @Author   : Daishijun
# @Site     : 
# @File     : prac1.py
# @Software : PyCharm

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from deepctr.models import xDeepFM, DeepFM, DCN
from deepctr.inputs import SparseFeat, DenseFeat, get_fixlen_feature_names





data = pd.read_csv('./input/final_track2_train.txt', sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'device', 'time', 'duration_time'])
sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'device', 'like']
dense_features = ['time', 'duration_time']

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0,)

target = ['finish']