from model import Cnn_baseline,Ann_Baseline
from utils import read,ann_preprocessing

X, Y,X_test,Y_test = ann_preprocessing()
# cnn_baseline = Cnn_baseline()
# cnn_baseline.fit(X,Y,X_test,Y_test,epochs=100)
# print(X_test[0].shape)
# cnn_baseline.predict(X_test[0])
# # ann_baseline = Ann_Baseline()
# # ann_baseline.predict(X_test=X_test[0])
# cnn_baseline.export_tf_serving()


ann_Baseline = Ann_Baseline()
ann_Baseline.fit(X,Y,X_test,Y_test,epochs=100)
ann_Baseline.predict(X_test)
# ann_baseline = Ann_Baseline()
# ann_baseline.predict(X_test=X_test[0])
ann_Baseline.export_tf_serving()


