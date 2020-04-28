# @Time : 2020/4/27 15:09 
# @Author : Jinguo Zhu
# @File : PCA_t-SNE.py 
# @Software: PyCharm
'''

 '''
Class_name = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]

import numpy as np
import pandas as pd

Save_Path = '../driftdataset/batch{}.csv'
datatrain = []
for i in range(1, 11):
    datatrain.append(pd.read_csv(Save_Path.format(i), header=None))
# ### Change dataframe to array
X = [np.array(data) for data in datatrain]
datatrain_array=np.concatenate(X) # [13910, 130]
from sklearn.preprocessing import MaxAbsScaler
xtrain = datatrain_array[:,2:130]
ytrain = datatrain_array[:,0]

max_abs_scaler = MaxAbsScaler()
xtrain = max_abs_scaler.fit_transform(xtrain)

# ### PCA plot
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
xtrain=pca.fit_transform(xtrain)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 11
for i, gas in enumerate(Class_name):
    ax.plot(xtrain[ytrain==(i+1),0], xtrain[ytrain==(i+1),1],  'o', markersize=2.5, label=gas)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend(loc='upper right')
plt.show()
# ### t-SNE plot

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
tsne= TSNE(n_components=2,n_iter=1000)#change perplexity for better result
xtrain=tsne.fit_transform(xtrain)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 11
for i, gas in enumerate(Class_name):
    ax.plot(xtrain[ytrain==(i+1),0], xtrain[ytrain==(i+1),1],  'o', markersize=2.5, label=gas)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend(loc='upper right')
plt.show()