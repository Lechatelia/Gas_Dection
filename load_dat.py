# @Time : 2020/4/26 23:54 
# @Author : Jinguo Zhu
# @File : load_dat.py 
# @Software: PyCharm
'''
demo for load dat files
 '''

import numpy as np
import pandas as pd

Root = '../driftdataset/batch{}.dat'
Save_Path = '../driftdataset/batch{}.csv'

for i in range(1,11):
    f=open(Root.format(i))
    sentimentlist = []
    for line in f:
        s = line.strip().split(' ')
        s1 = s[0].split(';') # 气体种类与浓度
        s2 = [ss.split(':')[-1] for ss in s[1:]] # 气体特征 128维
        s = list(map(float, s1+s2))
        sentimentlist.append(s)
    batch_data = np.array(sentimentlist)

    pf = pd.DataFrame(data=batch_data)
    pf.to_csv(Save_Path.format(i), header=False, index=False)

    # 测试读取是否成功
    # datatrain1=np.array(pd.read_csv('../dataset/Multi-class Dataset/Batch{}e.csv'.format(i), header=None))
    # print( (batch_data-datatrain1).sum())


