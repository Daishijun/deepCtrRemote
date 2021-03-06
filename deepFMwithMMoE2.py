#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/22 11:26
# @Author   : Daishijun
# @Site     : 
# @File     : deepFMwithMMoE2.py
# @Software : PyCharm

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/21 3:56
# @Author   : Daishijun
# @Site     :
# @File     : deepFMwithMMoE.py
# @Software : PyCharm


"""
     class deepFM
"""

import tensorflow as tf

from deepctr.inputs import input_from_feature_columns, get_linear_logit, build_input_features, combined_dnn_input
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import FM
from deepctr.layers.utils import concat_fun
from tensorflow.python.keras.layers import Dense

from mmoe import MMoE

def DeepFM(linear_feature_columns, dnn_feature_columns, embedding_size=8, use_fm=True, only_dnn=False, dnn_hidden_units=(128, 128),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param embedding_size: positive integer,sparse feature embedding_size
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    # ## 为每个特征创建Input[1,]； feature == > {'feature1': Input[1,], ...}
    # features = build_input_features(linear_feature_columns + dnn_feature_columns)
    #
    # ## [Input1, Input2, ... ]
    # inputs_list = list(features.values())
    #
    # sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
    #                                                                      embedding_size,
    #                                                                      l2_reg_embedding, init_std,
    #                                                                      seed)
    # ## [feature_1对应的embedding层，下连接对应feature1的Input[1,]层,...], [feature_1对应的Input[1,]层,...]
    #
    # linear_logit = get_linear_logit(features, linear_feature_columns, l2_reg=l2_reg_linear, init_std=init_std,
    #                                 seed=seed, prefix='linear')
    # ## 线性变换层，没有激活函数
    #
    # fm_input = concat_fun(sparse_embedding_list, axis=1)
    # ## 稀疏embedding层concate在一起
    #
    # fm_logit = FM()(fm_input)
    # ## FM的二次项部分输出，不包含一次项和bias
    #
    # dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    #
    # # dnn_out = Dense(128, dnn_activation, l2_reg_dnn, dnn_dropout,
    # #               dnn_use_bn, seed)(dnn_input)
    #
    # dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
    #               dnn_use_bn, seed)(dnn_input)
    #
    #
    # finish_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed)(dnn_out)
    # finish_logit = tf.keras.layers.Dense(1,use_bias=False, activation=None )(finish_out)
    #
    # like_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed)(dnn_out)
    # like_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(like_out)
    #
    #
    # dnn_logit = tf.keras.layers.Dense(
    #     1, use_bias=False, activation=None)(dnn_out)
    # # if len(dnn_hidden_units) > 0 and only_dnn == True:
    # #     final_logit = dnn_logit
    # # elif len(dnn_hidden_units) == 0 and use_fm == False:  # only linear
    # #     final_logit = linear_logit
    # # elif len(dnn_hidden_units) == 0 and use_fm == True:  # linear + FM
    # #     final_logit = tf.keras.layers.add([linear_logit, fm_logit])
    # # elif len(dnn_hidden_units) > 0 and use_fm == False:  # linear +　Deep
    # #     final_logit = tf.keras.layers.add([linear_logit, dnn_logit])
    # # elif len(dnn_hidden_units) > 0 and use_fm == True:  # linear + FM + Deep
    # #     final_logit = tf.keras.layers.add([linear_logit, fm_logit, dnn_logit])
    # # else:
    # #     raise NotImplementedError
    #
    # finish_logit = tf.keras.layers.add([linear_logit, fm_logit, finish_logit])
    # like_logit = tf.keras.layers.add([linear_logit, fm_logit, like_logit])
    #
    #
    #
    # output_finish = PredictionLayer('binary', name='finish')(finish_logit)
    # output_like = PredictionLayer('binary', name='like')(like_logit)
    # model = tf.keras.models.Model(inputs=inputs_list, outputs=[output_finish, output_like])
    # return model




    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features,dnn_feature_columns,
                                                                              embedding_size,
                                                                              l2_reg_embedding,init_std,
                                                                              seed)

    linear_logit = get_linear_logit(features, linear_feature_columns, l2_reg=l2_reg_linear, init_std=init_std,
                                    seed=seed, prefix='linear')

    fm_input = concat_fun(sparse_embedding_list, axis=1)
    fm_logit = FM()(fm_input)

    dnn_input = combined_dnn_input(sparse_embedding_list,dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                   dnn_use_bn, seed)(dnn_input)

    finish_out = DNN(dnn_hidden_units)(dnn_out)
    finish_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(finish_out)

    like_out = DNN(dnn_hidden_units)(dnn_out)
    like_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(like_out)

    finish_logit = tf.keras.layers.add(
        [linear_logit, finish_logit, fm_logit])
    like_logit = tf.keras.layers.add(
        [linear_logit, like_logit, fm_logit])

    finish_output = PredictionLayer(task, name="finish_output")(finish_logit)
    like_output =  PredictionLayer(task, name="like_output")(like_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=[finish_output, like_output])
    return model

