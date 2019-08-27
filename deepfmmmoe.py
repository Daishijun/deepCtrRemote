import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
#from deepctr.models import xDeepFM, DeepFM, DCN
from train2modelmmoe import DeepFMmmoe
from deepctr.inputs import SparseFeat, DenseFeat, get_fixlen_feature_names
from tensorflow.python.keras.optimizers import Adam,Adagrad
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    DATA_PATH = '/opt/ByteCamp/'
    DATA_FILE = 'bytecamp.data'

    data = pd.read_csv(DATA_PATH + DATA_FILE, sep=',')
    sparse_features = ['uid', 'u_region_id', 'item_id', 'author_id', 'music_id']
    dense_features = ['duration', 'generate_time']

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    target = ['finish', 'like']
    # target = ['finish']

    data['generate_time'] %= 60 * 60 * 24

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    sparse_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              # (特征名, 特征不同取值个数)生成SparseFeat对象，name == 特征名，dimension==该特征不同取值个数， dtype ==int32
                              for feat in sparse_features]
    dense_feature_columns = [DenseFeat(feat, 1)  # （特征名， dimension==1） 数据dtype == float32
                             for feat in dense_features]
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns
    linear_feature_columns = sparse_feature_columns + dense_feature_columns

    ## 这里有多余的步骤，该方法中间为每个特征设置了Input层，但是没有返回，只返回了特征名称list，其实可以直接从上面的两个list合并得到。
    feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)

    RIGIONID = 0
    train_indexs = data[(data['date'] < 20190708) & (data['u_region_id'] == RIGIONID)].index

    test_indexs = data[(data['date'] == 20190708) & (data['u_region_id'] == RIGIONID)].index

    train, test = data.loc[train_indexs], data.loc[test_indexs]

    train_model_input = [train[name] for name in feature_names]

    test_model_input = [test[name] for name in feature_names]
    #test_labels = [test[target[0]].values, test[target[1]].values]
    # model = xDeepFM(linear_feature_columns, dnn_feature_columns, embedding_size=8, dnn_hidden_units=(256, 256),
    #                 cin_layer_size=(256, 256,), cin_split_half=True, cin_activation='relu', l2_reg_linear=0.00001,
    #                 l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, init_std=0.0001, seed=1024, dnn_dropout=0,
    #                 dnn_activation='relu', dnn_use_bn=False, task='binary')

    model = DeepFMmmoe(linear_feature_columns, dnn_feature_columns, embedding_size=8, use_fm=True, dnn_hidden_units=(128, 128,),
           l2_reg_linear=0.0001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary')
    # print("Total Parameters：%d" % model.count_params())
    # input()
    # model = DCN(dnn_feature_columns, embedding_size=8, cross_num=5, dnn_hidden_units=(512,256,128,64,32), l2_reg_embedding=1e-5,
    #     l2_reg_cross=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_use_bn=False,
    #     dnn_activation='relu', task='binary')
    model.summary()
    def auroc(y_true, y_pred):
        return tf.numpy_function(roc_auc_score, (y_true, y_pred), tf.double)
    model.compile(Adam(lr=0.0001), "binary_crossentropy", metrics=[auroc], loss_weights=[0.6,0.4],)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    # history = model.fit(train_model_input, {"finish":train["finish"].values, "like":train["like"].values},
    #                     batch_size=4096, epochs=20, verbose=1, validation_split=0.1, callbacks=[early_stopping])
    history = model.fit(train_model_input, {"finish": train["finish"].values, "like": train["like"].values},
                        batch_size=4096, epochs=10, verbose=1, validation_split=0.2,
                        callbacks=[EarlyStopping(monitor='val_finish_auc', patience=2, verbose=1
                                                 )])

    pred_ans = model.predict(test_model_input, batch_size=2 ** 14)
    for i in range(len(target)):
        print("=={}==".format(target[i]))
        pred_finish = (pred_ans[i] * 2).astype(int)
        print("test accuracy", round(accuracy_score(test[target[i]].values, pred_finish), 4))
        print("test LogLoss", round(log_loss(test[target[i]].values, pred_ans[i]), 4))
        print("test AUC", round(roc_auc_score(test[target[i]].values, pred_ans[i]), 4))
