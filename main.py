# @Time : 2020/4/27 10:38 
# @Author : Jinguo Zhu
# @File : main.py 
# @Software: PyCharm
'''
this script used for evaluate machine learning methods
 '''

from GasNet import *
import argparse
import numpy as py
import pandas


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector for gas detection')
    parser.add_argument('--method', default='nerual_network', type=str, help='the method',
                        choices=['KNN', "AdaBoost","LogisticRegression", "DecisionTree", "ExtraTrees",
                                 "KNeighbors", "Majority_Voting", "Naive_Bayes", "RandomForest", "LinearSVC",
                                 "nerual_network"])
    parser.add_argument('--dataset', default='../driftdataset/batch{}.csv', type=str, help='the dataset prefix')
    parser.add_argument('--output', default='output', type=str, help='path to save imgs')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    Algorithm_name = args.method
    print("the algorithm is {}".format(Algorithm_name))
    x_train, x_test, y_train, y_test = Read_dataset(args.dataset, Algorithm_name)
    Algorithm = eval(Algorithm_name)
    y_pred, acc = Algorithm(x_train, y_train, x_test, y_test)
    print('ground truth           :', y_test)
    print('predicted class        :', y_pred)
    print('cross validation acc   :', acc)
    Confusion_matrix(y_test, y_pred, Algorithm_name, args.output)


if __name__ == "__main__":
    main()
