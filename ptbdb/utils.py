import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read():
    df_normal = pd.read_csv("input/ptbdb_normal.csv", header=None)
    df_abnormal = pd.read_csv("input/ptbdb_abnormal.csv", header=None)
    X = pd.concat([df_normal, df_abnormal])
    Y = X[X.columns[-1]]

    X = X.iloc[:, :-1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337, stratify=Y)

    X_train = X_train.to_numpy().reshape(len(X_train), X_train.shape[1], 1)
    X_test = X_test.to_numpy().reshape(len(X_test), X_test.shape[1],1)

    return X_train, Y_train, X_test, Y_test

def ann_preprocessing():
    df_normal = pd.read_csv("input/ptbdb_normal.csv", header=None)
    df_abnormal = pd.read_csv("input/ptbdb_abnormal.csv", header=None)
    X = pd.concat([df_normal, df_abnormal])
    Y = X[X.columns[-1]]

    X = X.iloc[:, :-1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337, stratify=Y)

    # X_train = X_train.to_numpy().reshape(len(X_train), X_train.shape[1], 1)
    # X_test = X_test.to_numpy().reshape(len(X_test), X_test.shape[1],1)

    return X_train, Y_train, X_test, Y_test
