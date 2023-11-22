import pandas as pd
import numpy as np

from keras.models import Sequential
from keras import optimizers, losses, activations
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf

class Cnn_baseline:
    def __init__(self) -> None:
        self.net = Sequential([
            Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid", input_shape=(187, 1)),
            Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid"),
            MaxPool1D(pool_size=2),
            Dropout(rate=0.1),
            Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"),
            Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"),
            MaxPool1D(pool_size=2),
            Dropout(rate=0.1),
            Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"),
            Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"),
            MaxPool1D(pool_size=2),
            Dropout(rate=0.1),
            Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid"),
            Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid"),
            GlobalMaxPool1D(),
            Dropout(rate=0.2),
            Dense(64, activation=activations.relu),
            Dense(64, activation=activations.relu),
            Dense(1, activation=activations.sigmoid)
        ])
        opt = optimizers.Adam(0.001)
        self.net.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
        self.net.summary()
        file_path = "baseline_cnn_ptbdb.h5"
    
        checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
        redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
        self.callbacks_list = [checkpoint, early, redonplat]

    def fit(self,X,Y,epochs):
        self.net.fit(X, Y, epochs=epochs, verbose=2, callbacks=self.callbacks_list, validation_split=0.1)
        
    def predict(self,X_test):
        pred_test = self.net.predict(X_test)
        pred_test = (pred_test>0.5).astype(np.int8)
    
    def eval(self,X_test,Y_test):
        pred_test = self.net.predict(X_test)
        f1 = f1_score(Y_test, pred_test)
        print("Test f1 score : %s "% f1)
        acc = accuracy_score(Y_test, pred_test)
        print("Test accuracy score : %s "% acc)
    
    def save_h5(self,filepath:str="baseline_cnn_ptbdb.h5"):
        self.net.save_weights(filepath)
    
    def export_tf_serving(self,path:str='export_tf/ptpdb/cnn_baseline'):
        tf.saved_model.save(self.net, path)