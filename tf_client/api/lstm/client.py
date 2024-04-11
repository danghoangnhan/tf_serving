import logging
from flask import request
from flask_restx import Resource
import numpy as np
import pandas as pd
import tensorflow as tf
from api.restplus import api
from werkzeug.datastructures import FileStorage
from tf_client.api.logic.tf_serving_client import make_prediction


# create dedicated namespace for LSTM client
lstm_namespace = api.namespace('rhm', description='Operations for LSTM client')

# Flask-RestPlus specific parser for image uploading
UPLOAD_KEY = 'image'
UPLOAD_LOCATION = 'files'
upload_parser = api.parser()
upload_parser.add_argument(UPLOAD_KEY,
                           location=UPLOAD_LOCATION,
                           type=FileStorage,
                           required=True)


def normalize_series(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


def create_windowed_dataset(series, n_past=24, n_future=24, shift=1, batch_size=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past][np.newaxis, ...], w[n_past:][np.newaxis, ...]))  # Add batch dimension
    return ds.batch(batch_size).prefetch(1)


def preprocess_data(data):
    df = pd.DataFrame(data)
    data_np = df.values
    normalized_data = normalize_series(data_np, data_np.min(axis=0), data_np.max(axis=0))
    windowed_dataset = create_windowed_dataset(normalized_data)
    return tf.data.experimental.get_single_element(windowed_dataset)



@lstm_namespace.route('/prediction')
class Prediction(Resource):
    @lstm_namespace.doc(description='Predict the using LSTM model. ' +
            'Return 3 most probable digits with their probabilities',
            responses={
                200: "Success",
                400: "Bad request",
                500: "Internal server error"
                })
    
    @lstm_namespace.expect(model_input, validate=True)
    def post(self):
        try:
            sequence_data = request.json['sequence']
            preprocessed_data = preprocess_data(sequence_data)
            results = make_prediction(preprocessed_data)
            return {'prediction_result': results}, 200
        except Exception as inst:
            logging.error(f"Error processing prediction request: {inst}")
            return {'message': 'Internal error: {}'.format(inst)}, 500

