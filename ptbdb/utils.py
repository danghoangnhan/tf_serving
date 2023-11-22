import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read():
    df_normal = pd.read_csv("input/ptbdb_normal.csv", header=None)
    df_abnormal = pd.read_csv("input/ptbdb_abnormal.csv", header=None)
    df = pd.concat([df_normal, df_abnormal])
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])
    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]
    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
    return X, Y,X_test,Y_test
