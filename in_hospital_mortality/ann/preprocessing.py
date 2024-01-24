import tensorflow as tf
from tensorflow.keras.layers import Layer

class CustomScalingLayer(Layer):
    def __init__(self, minmax_indices, zscore_indices, minmax_params, zscore_params):
        super(CustomScalingLayer, self).__init__()
        # Indices of columns to be MinMax scaled and Z-Score normalized
        self.minmax_indices = minmax_indices
        self.zscore_indices = zscore_indices

        # Parameters for MinMax and Z-Score (e.g., min, max, mean, std)
        self.minmax_params = minmax_params
        self.zscore_params = zscore_params

    def call(self, inputs):
        # Apply MinMax scaling
        for i, (min_val, max_val) in zip(self.minmax_indices, self.minmax_params):
            inputs[:, i] = (inputs[:, i] - min_val) / (max_val - min_val)

        # Apply Z-Score scaling
        for i, (mean_val, std_val) in zip(self.zscore_indices, self.zscore_params):
            inputs[:, i] = (inputs[:, i] - mean_val) / std_val

        return inputs
