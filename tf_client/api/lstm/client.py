import logging
from flask import request
from flask_restx import Resource
from api.restplus import api
from werkzeug.datastructures import FileStorage
from tf_client.api.logic.tf_serving_client import make_prediction
from tf_client.settings import ModelConfig
from utils import preprocessing
from utils.requestLoader import InHospitalMortalityRequestReader
from utils.preprocessing import Discretizer, Normalizer
import os

fields =  [
    "Hours", 
    "Capillary refill rate", 
    "Diastolic blood pressure", 
    "Fraction inspired oxygen", 
    "Glascow coma scale eye opening", 
    "Glascow coma scale motor response", 
    "Glascow coma scale total", 
    "Glascow coma scale verbal response", 
    "Glucose", 
    "Heart Rate", 
    "Height", 
    "Mean blood pressure", 
    "Oxygen saturation", 
    "Respiratory rate", 
    "Systolic blood pressure", 
    "Temperature", 
    "Weight", 
    "pH"
  ]

normalize_list = [
    'ihm_ts1.0.input_str:previous.start_time:zero',
    'ihm_ts2.0.input_str:previous.start_time:zero',
    'ihm_ts0.8.input_str:previous.start_time:zero'
]

parameter_config = {
    'dim': 16, 
    'timestep' :1.0,
    'depth': 2,
    'dropout': 0.3,
    'mode':'test'
    'batch_size': 8 
}--load_state in_hospital_mortality/keras_states/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch27.test0.27868239298546116.state

# create dedicated namespace for LSTM client
rhm_namespace = api.namespace('rhm', description='Operations for rhm client')

RNNModelConfig = ModelConfig(
    MODEL_NAME = 'rnn',
    MODEL_SIGNATURE_NAME = 'predict_time_series',
    MODEL_INPUTS_KEY = 'conv2d_input'
)  



def normalize_series(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def preprocess_data(data,fields,normalizer_state,timestep,small_part):
    
    data_reader = InHospitalMortalityRequestReader(data)
    discretizer = Discretizer(timestep=timestep,
                            store_masks=True,
                            impute_strategy='previous',
                            start_time='zero')

    normalizer = Normalizer(fields=fields) 

    if normalizer_state is None:
        normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)

    ret = preprocessing.load_data(data_reader, 
                                discretizer, 
                                normalizer,
                                small_part,
                                return_names=True)
    
    data = ret["data"][0]
    labels = ret["data"][1]
    return data, labels
    
@rhm_namespace.route('/prediction')
class Prediction(Resource):
    @rhm_namespace.doc(description='Predict the using LSTM model. ' +
            'Return 3 most probable digits with their probabilities',
            responses={
                200: "Success",
                400: "Bad request",
                500: "Internal server error"
                })
    
    def post(self):
        try:
            fields = request.json['fields']
            data = request.json['data']
            preprocessed_data = preprocess_data(
                data=data,
                fields=fields,
                normalizer_state=None
            )
            results = make_prediction(preprocessed_data)
            results = make_prediction(data)
            return {'prediction_result': results}, 200
        
        except Exception as inst:
            logging.error(f"Error processing prediction request: {inst}")
            return {
                'message': 'Internal error: {}'.format(inst)}, 500

