import numpy as np

from keras.models import Sequential
from keras import optimizers, losses, activations
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
import time
import os

class Cnn_baseline:
    def __init__(self) -> None:
        self.net = Sequential([
            Convolution1D(16, kernel_size=5, activation=activations.relu, padding="same", input_shape=(187,1)),
            Convolution1D(16, kernel_size=5, activation=activations.relu, padding="same"),
            MaxPool1D(pool_size=2, padding="same"),
            Dropout(rate=0.1),
            Convolution1D(32, kernel_size=3, activation=activations.relu, padding="same"),
            Convolution1D(32, kernel_size=3, activation=activations.relu, padding="same"),
            MaxPool1D(pool_size=2, padding="same"),
            Dropout(rate=0.1),
            Convolution1D(32, kernel_size=3, activation=activations.relu, padding="same"),
            Convolution1D(32, kernel_size=3, activation=activations.relu, padding="same"),
            MaxPool1D(pool_size=2, padding="same"),
            Dropout(rate=0.1),
            Convolution1D(256, kernel_size=3, activation=activations.relu, padding="same"),
            Convolution1D(256, kernel_size=3, activation=activations.relu, padding="same"),
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

    def fit(self,X,Y,X_test,Y_test,epochs):
        history = self.net.fit(
        X, Y,
        epochs=epochs,
        verbose=2,
        callbacks=self.callbacks_list,
        validation_data=(X_test, Y_test)
    )
        return history
    
    def predict(self,X_test):
        pred_test = self.net.predict(X_test)
        pred_test = (pred_test>0.5).astype(np.int8)
        
        return pred_test
    def eval(self,X_test,Y_test):
        pred_test = self.net.predict(X_test)
        f1 = f1_score(Y_test, pred_test)
        print("Test f1 score : %s "% f1)
        acc = accuracy_score(Y_test, pred_test)
        print("Test accuracy score : %s "% acc)
    
    def save_h5(self,filepath:str="baseline_cnn_ptbdb.h5"):
        self.net.save_weights(filepath)
    
    
    def export_tf_serving(self,MODEL_DIR:str='export_tf/ptpdb/cnn_baseline',version :int= 1):
        export_path = os.path.join(MODEL_DIR, str(version))
        tf.keras.models.save_model(
            self.net,
            export_path,
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )


class Ann_Baseline:
    def __init__(self) -> None:
        self.net = Sequential([
            # First dense layer with 16 units
            Dense(16, activation='relu', input_shape=(187,)),  # Assuming input shape is 21
            # Second dense layer with 16 units
            Dense(16, activation='relu'),
            # Third dense layer with 256 units
            Dense(256, activation='relu'),
            # Dropout layer with a dropout rate of 0.5
            Dropout(0.5),
            # Output layer with a single unit
            Dense(1, activation='sigmoid')  # or 'linear' depending on your problem
        ])
        opt = optimizers.Adam(0.001)
        self.net.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
        self.net.summary()
        file_path = "baseline_cnn_ptbdb.h5"
    
        checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
        redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
        self.callbacks_list = [checkpoint, early, redonplat]

    def fit(self,X,Y,X_test,Y_test,epochs):
        self.net.fit(X, Y, epochs=epochs, verbose=2, callbacks=self.callbacks_list)
        
    def predict(self,X_test):
        pred_test = self.net.predict(X_test)
        pred_test = (pred_test>0.5).astype(np.int8)
        return pred_test
    
    def eval(self,X_test,Y_test):
        pred_test = self.net.predict(X_test)
        f1 = f1_score(Y_test, pred_test)
        print("Test f1 score : %s "% f1)
        acc = accuracy_score(Y_test, pred_test)
        print("Test accuracy score : %s "% acc)
    
    def save_h5(self,filepath:str="baseline_cnn_ptbdb.h5"):
        self.net.save_weights(filepath)
    
    
    def export_tf_serving(self,MODEL_DIR:str='export_tf/ptpdb/ann_baseline',version :int= 1):
        export_path = os.path.join(MODEL_DIR, str(version))
        tf.keras.models.save_model(
            self.net,
            export_path,
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )