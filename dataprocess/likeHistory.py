

import pandas as pd
import os
import numpy as np

DATA_FILE= '/opt/ByteCamp/bytecamp.data'

# DATA_FILE = r'D:\Users\admin\PycharmProjects\pracproject\file.csv'

data = pd.read_csv(DATA_FILE, sep=',').head(1000000)
# sparse_features = ['uid', 'u_region_id', 'item_id', 'author_id','music_id']
# dense_features = ['duration', 'generate_time']

# data[sparse_features] = data[sparse_features].fillna('-1', )
# data[dense_features] = data[dense_features].fillna(0,)

def getlikerate(row, targetID):
    tmpdata = data.loc[:row+1,:]
    # print('tmp data:', tmpdata.to_dense())
    # print(tmpdata.columns)
    tmpdata = tmpdata[tmpdata['uid']==targetID]
    likecounts = tmpdata['like'].value_counts()
    print(likecounts)
    # like_1_num = likecounts[1] if 1 in likecounts.index() else 0
    like_1_num = 0
    # print('likecounts index:', likecounts.index, list(likecounts.index))
    # if len(likecounts)>1:
    #     print('like 1:', likecounts[1])

    try:
        like_1_num = likecounts[1] if 1 in list(likecounts.index) else 0
        print('like 1 num:',like_1_num)
        # print('likecount:', likecounts)
    except:
        pass


    return np.float(like_1_num/likecounts.sum())
    # return 99

print('read ok')

for i, _ in data.iterrows():
    data.loc[i, 'uid_like_rate'] = getlikerate(i,1)

print('data:')
print(data.to_dense())



