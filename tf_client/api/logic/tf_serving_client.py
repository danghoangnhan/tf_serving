from __future__ import print_function

import operator
import logging
import settings
import tensorflow as tf

# Communication to TensorFlow server via gRPC
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
FLAGS = tf.compat.v1.app.flags.FLAGS

tf.compat.v1.app.flags.DEFINE_string('server', 'tf-serving-server:8500','PredictionService host:port')
tf.compat.v1.app.flags.DEFINE_string('image', '','path to image in JPEG format')

log = logging.getLogger(__name__)

def __get_tf_server_connection_params__(tfServerConfig:settings.TFServerconfig):
    '''
    Returns connection parameters to TensorFlow Server
    :return: Tuple of TF server name and server port
    '''
    server_name = settings.get_env_var_setting('TF_SERVER_NAME', tfServerConfig.DEFAULT_TF_SERVER_NAME)
    server_port = settings.get_env_var_setting('TF_SERVER_PORT', tfServerConfig.DEFAULT_TF_SERVER_PORT)
    return server_name, server_port

def __create_prediction_request__(modelConfig:settings.ModelConfig,image):
    '''
    Creates prediction request to TensorFlow server for CNN model
    :param: Byte array, image for prediction
    :return: PredictRequest object
    '''
    request = predict_pb2.PredictRequest()
    request.model_spec.name = modelConfig.MODEL_NAME
    request.model_spec.signature_name = modelConfig.MODEL_SIGNATURE_NAME
    request.inputs[modelConfig.MODEL_INPUTS_KEY].CopyFrom(tf.make_tensor_proto(image))
    return request

def __open_tf_server_channel__(server_name, server_port)->prediction_service_pb2_grpc.PredictionServiceStub:
    '''
    Opens channel to TensorFlow server for requests

    :param server_name: String, server name (localhost, IP address)
    :param server_port: String, server port
    :return: Channel stub
    '''
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub

def __make_prediction_and_prepare_results__(stub:prediction_service_pb2_grpc.PredictionServiceStub, request):
    '''
    Sends Predict request over a channel stub to TensorFlow server
    :param stub: Channel stub
    :param request: PredictRequest object
    :return: List of tuples, 3 most probable digits with their probabilities
    '''
    result = stub.Predict(request, 60.0)  # 60 secs timeout
    probs = result.outputs['dense_1'].float_val
    value_dict = {idx: prob for idx, prob in enumerate(probs)}
    sorted_values = sorted(
        value_dict.items(),
        key=operator.itemgetter(1),
        reverse=True)
    return sorted_values

def make_prediction(modelConfig:settings.ModelConfig,image):
    '''
    Predict the house number on the image using CNN model
    :param image: Byte array, images for prediction
    :return: List of tuples, 3 most probable digits with their probabilities
    '''
    # get TensorFlow server connection parameters
    server_name, server_port = __get_tf_server_connection_params__(modelConfig)
    log.info('Connecting to TensorFlow server %s:%s', server_name, server_port)
    stub = __open_tf_server_channel__(server_name, server_port)
    log.info("stub is created")
    request = __create_prediction_request__(modelConfig,image)
    return __make_prediction_and_prepare_results__(stub, request)