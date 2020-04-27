# @Time : 2020/4/27 10:09 
# @Author : Jinguo Zhu
# @File : utils.py 
# @Software: PyCharm
'''
this is used for pre or post processing
 '''
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.xlim(-0.5, len(classes)-0.5)
    plt.ylim(-0.5, len(classes)-0.5)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def Read_dataset(Save_Path, algorithm_name):
    # Save_Path = '../driftdataset/batch{}.csv'
    datatrain = []
    for i in range(1, 11):
        datatrain.append(pd.read_csv(Save_Path.format(i), header=None))
    # 注意原本不存在列索引

    # ### Change dataframe to array
    X = [np.array(data) for data in datatrain]

    # sample = np.concatenate([X[0], X[1]])
    # lengths = [len(X[0]), len(X[1])]
    datatrain_array=np.concatenate(X) # [13910, 130]


    # ### Split x and y (feature and target)

    xtrain = datatrain_array[:,2:130] #[num, 128]
    ytrain = datatrain_array[:,0] # [num]
    #数据归一化
    if algorithm_name == "Naive_Bayes":
        min_max_scaler = MinMaxScaler()
        xtrain = min_max_scaler.fit_transform(xtrain)
    else:
        max_abs_scaler = MaxAbsScaler()
        xtrain = max_abs_scaler.fit_transform(xtrain)
# ### Train and test split

    # X_train, X_test, y_train, y_test = \
    return train_test_split(xtrain, ytrain, test_size=.1,random_state=1)

def Confusion_matrix(y_test, y_pred, Algorithm, output):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    class_names=['1','2','3','4','5','6']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    rootdir = os.path.join(output, Algorithm +'_1.png')
    plt.savefig(rootdir,figsize=(4,3),dpi=500,bbox_inches='tight',labelsize=12)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Confusion matrix, with normalization')
    plt.legend(['train', 'test'], loc='lower right')
    rootdir = os.path.join(output, Algorithm +'_2.png')
    plt.savefig(rootdir,figsize=(4,3),dpi=500,bbox_inches='tight',labelsize=11)
    plt.show()

def max_scalar(x):
    max_abs_scaler = MaxAbsScaler()
    return max_abs_scaler.fit_transform(x)

def min_max_scalar(x):
    max_abs_scaler = MinMaxScaler()
    return max_abs_scaler.fit_transform(x)