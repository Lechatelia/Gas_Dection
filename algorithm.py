# @Time : 2020/4/27 11:10 
# @Author : Jinguo Zhu
# @File : algorithm.py 
# @Software: PyCharm
'''
the repo for different algorithms
 '''
from GasNet import *

def KNN(X_train, y_train, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.model_selection import cross_val_score
    bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=14),
                                max_samples=0.5, max_features=0.5)
    bagging=bagging.fit(X_train,y_train)

    y_pred =bagging.predict(X_test)
    return y_pred, cross_val_score(bagging,X_test,y_test).mean()

def AdaBoost(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import cross_val_score
    ada=AdaBoostClassifier(n_estimators=10)
    ada=ada.fit(X_train,y_train)
    y_pred =ada.predict(X_test)
    return y_pred, cross_val_score(ada,X_test,y_test).mean()


def LogisticRegression(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression()
    log=clf.fit(X_train,y_train)
    y_pred =log.predict(X_test)
    return y_pred, cross_val_score(log,X_test,y_test).mean()


def DecisionTree(X_train, y_train, X_test, y_test):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score

    clf = DecisionTreeClassifier()
    tree=clf.fit(X_train,y_train)

    y_pred =tree.predict(X_test)
    return y_pred, cross_val_score(tree,X_test,y_test).mean()


def ExtraTrees(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import cross_val_score

    extrarandom=ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
    extrarandom=extrarandom.fit(X_train,y_train)

    y_pred =extrarandom.predict(X_test)
    return y_pred, cross_val_score(extrarandom,X_test,y_test).mean()


def KNeighbors(X_train, y_train, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    clf = KNeighborsClassifier(n_neighbors=30)
    neigh=clf.fit(X_train,y_train)

    y_pred =neigh.predict(X_test)
    return y_pred, cross_val_score(neigh,X_test,y_test).mean()

def Majority_Voting(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.model_selection import cross_val_score

    clf1 = LogisticRegression()
    clf2= DecisionTreeClassifier()
    clf3= LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                    intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                    multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                    verbose=0)
    clf4= KNeighborsClassifier(n_neighbors=30)
    eclf = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('svm', clf3),('knn',clf4)], voting='hard')
    eclf = eclf.fit(X_train,y_train)
    y_pred =eclf.predict(X_test)
    return y_pred, cross_val_score(eclf,X_test,y_test).mean()


def Naive_Bayes(X_train, y_train, X_test, y_test):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    clf = MultinomialNB()
    gauss=clf.fit(X_train,y_train)
    y_pred =gauss.predict(X_test)
    return y_pred, cross_val_score(gauss,X_test,y_test).mean()

def RandomForest(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    clf = RandomForestClassifier()
    random=clf.fit(X_train,y_train)

    y_pred =random.predict(X_test)
    return y_pred, cross_val_score(random,X_test,y_test).mean()


def LinearSVC(X_train, y_train, X_test, y_test):
    from sklearn import svm
    from sklearn.model_selection import cross_val_score

    clf = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                        intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                        multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                        verbose=0)
    svm=clf.fit(X_train,y_train)

    y_pred =svm.predict(X_test)
    return y_pred, cross_val_score(svm,X_test,y_test).mean()

def nerual_network(X_train, y_train, X_test, y_test):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import np_utils
    from sklearn.model_selection import cross_val_score
    Y_train = np_utils.to_categorical(y_train)
    # Y_test=np_utils.to_categorical(y_test)


    model = Sequential()
    model.add(Dense(X_train.shape[-1], input_dim=X_train.shape[-1], init='uniform', activation='relu'))
    model.add(Dense(50, init='uniform', activation='relu'))
    model.add(Dense(80, init='uniform', activation='relu'))
    model.add(Dense(7, init='uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse','mae','mape','acc'])
    MLP=model.fit(X_train, Y_train, epochs=1000, batch_size=150000,validation_split=0.33, verbose=0)
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

    accuracy=np.array(MLP.history['val_acc'])[-1] # 得到验证集的精度
    print("train accuracy : {}".format(accuracy))
    y_pred = y_pred.argmax(axis=1)
    return y_pred, accuracy






