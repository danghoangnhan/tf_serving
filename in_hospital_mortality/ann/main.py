import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Constants
FILE_PATH = "input/data01.csv"
MODEL_SAVE_PATH = "baseline_cnn_ptbdb.h5"
SCALER_SAVE_PATH = 'scaler.pkl'

# Unwanted columns
unwanted_columns = [
    "group","ID","Urine output", "hematocrit", "RBC", "MCH", "MCHC", "MCV", "RDW", "Leucocyte",
    "Platelets", "Neutrophils", "Basophils", "Lymphocyte", "PT", "INR", "NT-proBNP",
    "Creatine kinase", "Creatinine", "Urea nitrogen", "glucose", "Blood potassium",
    "Blood sodium", "Blood calcium", "Chloride", "Anion gap", "Magnesium ion",
    "PH", "Bicarbonate", "Lactic acid", "PCO2", "EF"
]

# Data Preprocessing
def load_and_scale_data(file_path, columns_to_drop, scaler=None):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None, None

    df = df.drop(columns=columns_to_drop)
    
    df["BMI"].fillna(df["BMI"].mean(), inplace=True)
    df["heart rate"].fillna(df["heart rate"].mean(), inplace=True)
    df["Systolic blood pressure"].fillna(df["Systolic blood pressure"].mean(), inplace=True)
    df["Diastolic blood pressure"].fillna(df["Diastolic blood pressure"].mean(), inplace=True)
    df["Respiratory rate"].fillna(df["Respiratory rate"].mean(), inplace=True)
    df["temperature"].fillna(df["temperature"].mean(), inplace=True)
    df["SP O2"].fillna(df["SP O2"].mean(), inplace=True)
    df = df.dropna(how="any")

    if scaler is None:
        scaler = StandardScaler()
        df.iloc[:, 3:] = scaler.fit_transform(df.iloc[:, 3:])
    else:
        df.iloc[:, 3:] = scaler.transform(df.iloc[:, 3:])
    return df, scaler

class StandardScalerLayer(tf.keras.layers.Layer):
    def __init__(self, mean, scale, **kwargs):
        super(StandardScalerLayer, self).__init__(**kwargs)
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.scale = tf.constant(scale, dtype=tf.float32)

    def call(self, inputs):
        return (inputs - self.mean) / self.scale

class Ann_Baseline(tf.keras.Model):
    def __init__(self, input_shape, scaler, **kwargs):
        super(Ann_Baseline, self).__init__(**kwargs)
        self.standard_scaler_layer = StandardScalerLayer(scaler.mean_, scaler.scale_)

        self.dense_layers = [
            tf.keras.layers.Dense(16, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]

    def call(self, inputs):
        x = self.standard_scaler_layer(inputs)
        for layer in self.dense_layers:
            x = layer(x)
        return x
    
    def compile_model(self):
        self.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2):
        return self.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def save_h5(self, file_path=MODEL_SAVE_PATH):
        self.save_weights(file_path)
    
    def export_tf_serving(self, model_dir='export_tf/moritality/ann_baseline', version=1):
        export_path = os.path.join(model_dir, str(version))
        tf.keras.models.save_model(self, export_path, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None)

    def save_scaling(self, scaler, path=SCALER_SAVE_PATH):
        joblib.dump(scaler, path)

# Main script
def main():
    df_scaled, scaler = load_and_scale_data(FILE_PATH, unwanted_columns)
    X = df_scaled.drop(columns=['outcome'])
    y = df_scaled['outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    model = Ann_Baseline(input_shape=(X_train.shape[1],),scaler=scaler)
    model.compile_model()
    model.train_model(X_train, Y_train, epochs=1)
    model.export_tf_serving()
if __name__ == "__main__":
    main()
