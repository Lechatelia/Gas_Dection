# @Time : 2020/4/27 15:33 
# @Author : Jinguo Zhu
# @File : drift_exp.py 
# @Software: PyCharm
'''
this script use classifers to evaluate sensor drift
 '''

from GasNet import *
import argparse
import numpy as py
import pandas


def parse_args():
    parser = argparse.ArgumentParser(description='use classifers to evaluate sensor drift ')
    print(" now it is only supported nerual_network")
    parser.add_argument('--method', default='nerual_network', type=str, help='the method',
                        choices=['KNN', "AdaBoost","LogisticRegression", "DecisionTree", "ExtraTrees",
                                 "KNeighbors", "Majority_Voting", "Naive_Bayes", "RandomForest", "LinearSVC",
                                 "nerual_network"])
    parser.add_argument('--dataset', default='../driftdataset/batch{}.csv', type=str, help='the dataset prefix')
    parser.add_argument('--train_batch', default=[1,2], type=str, help='the batchs for train else all for val')
    parser.add_argument('--output', default='output', type=str, help='path to save imgs')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    Algorithm_name = args.method
    print("the algorithm is {}".format(Algorithm_name))
    xtrain, ytrain, batch_ids = this_Read_dataset(args.dataset, Algorithm_name)
    # Algorithm = eval(Algorithm_name)
    accuracy, ba_ids,  accs = this_nerual_network(xtrain, ytrain, batch_ids, args.train_batch)
    print('cross validation acc   :', accuracy)
    for id, acc in zip(ba_ids, accs):
        print("batch id: {}  acc: {}".format(id, acc))


def this_nerual_network(xtrain, ytrain, batch_ids, train_batch):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import np_utils
    from sklearn.model_selection import cross_val_score

    #train dataset
    # x_train = xtrain[]
    train_mask = np.expand_dims(batch_ids,1).repeat(len(train_batch),axis=1) ==np.expand_dims(np.array(train_batch),0).repeat(batch_ids.shape[0],axis=0)
    train_mask = train_mask.sum(axis=1).astype(np.bool)
    X_train = xtrain[train_mask]
    y_train = ytrain[train_mask]
    # one hot向量
    Y_train = np_utils.to_categorical(y_train)

    X_test = xtrain[np.logical_not(train_mask)]
    y_test = ytrain[np.logical_not(train_mask)]
    batch_ids_test = batch_ids[np.logical_not(train_mask)]
    # Y_test=np_utils.to_categorical(y_test)


    model = Sequential()
    model.add(Dense(X_train.shape[-1], input_dim=X_train.shape[-1], init='uniform', activation='relu'))
    model.add(Dense(50, init='uniform', activation='relu'))
    model.add(Dense(80, init='uniform', activation='relu'))
    model.add(Dense(7, init='uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse','mae','mape','acc'])
    MLP=model.fit(X_train, Y_train, shuffle=True, epochs=300, batch_size=100,validation_split=0.01, verbose=0)
    y_pred = model.predict(X_test)
    Flag_show = True
    if Flag_show:
        # summarize history for accuracy
        fig = plt.figure(figsize=(4,3))
        plt.plot(MLP.history['acc'])
        plt.plot(MLP.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')
        rootdir = 'mlp_acc.png'
        plt.savefig(rootdir,figsize=(4,3),dpi=500,bbox_inches='tight',labelsize=11)
        plt.show()

        # summarize history for loss
        fig = plt.figure(figsize=(4,3))
        plt.plot(MLP.history['mean_squared_error'])
        plt.plot(MLP.history['val_mean_squared_error'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        rootdir = 'mlp_loss.png'
        plt.savefig(rootdir,figsize=(4,3),dpi=500,bbox_inches='tight',labelsize=11)
        plt.show()

    accuracy1=np.array(MLP.history['val_acc'])[-1] # 得到验证集的精度
    y_pred = y_pred.argmax(axis=1)
    accuracy = (y_pred==y_test).mean()

    # accurary for every batch
    acc = []
    ba_ids = []
    for batch_index in range(1, 11):
        if batch_index in train_batch:
            continue
        ba_ids.append(batch_index)
        y_pred_batch = y_pred[batch_ids_test==batch_index]
        y_test_batch = y_test[batch_ids_test==batch_index]
        acc.append((y_pred_batch==y_test_batch).mean())
    if Flag_show:
        fig = plt.figure(figsize=(4,3))
        plt.xlim(0,11)
        plt.ylim(0,1)
        plt.plot(np.array(ba_ids), np.array(acc))
        for a, b in zip(ba_ids, acc):
            plt.text(a, b, "%0.1f"%(100*b), ha='center', va='bottom', fontsize=8)
        plt.savefig("acc_batch.jpg")
        plt.show()

    print("train accuracy : {}".format(accuracy))

    return accuracy, ba_ids,  acc

def this_Read_dataset(Save_Path, algorithm_name):
    # Save_Path = '../driftdataset/batch{}.csv'
    datatrain = []
    for i in range(1, 11):
        datatrain.append(pd.read_csv(Save_Path.format(i), header=None))
    # 注意原本不存在列索引
    # ### Change dataframe to array
    X = [np.array(data) for data in datatrain]
    batch_ids = [np.ones(data.shape[0])*(i+1) for i, data in enumerate(X)]
    # sample = np.concatenate([X[0], X[1]])
    # lengths = [len(X[0]), len(X[1])]
    datatrain_array=np.concatenate(X) # [13910, 130]
    batch_ids=np.concatenate(batch_ids) # [13910, 130]

    xtrain = datatrain_array[:,2:130] #[num, 128]
    ytrain = datatrain_array[:,0] # [num]
    #数据归一化
    if algorithm_name == "Naive_Bayes":
        min_max_scaler = MinMaxScaler()
        xtrain = min_max_scaler.fit_transform(xtrain)
    else:
        max_abs_scaler = MaxAbsScaler()
        xtrain = max_abs_scaler.fit_transform(xtrain)

    return xtrain, ytrain, batch_ids

if __name__ == "__main__":
    main()