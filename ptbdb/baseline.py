from model import Cnn_baseline
from utils import read

X, Y,X_test,Y_test = read()
cnn_baseline = Cnn_baseline()
cnn_baseline.fit(X,Y,epochs=100)
cnn_baseline.export_tf_serving()

