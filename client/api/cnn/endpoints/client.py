import io
import logging

from flask import request
from flask_restx import Resource
import numpy as np
from api.restplus import api
from api.cnn.logic.tf_serving_client import make_prediction
from werkzeug.datastructures import FileStorage
from PIL import Image


# create dedicated namespace for GAN client
ns = api.namespace('ecg', description='Operations for GAN client')

# Flask-RestPlus specific parser for image uploading
UPLOAD_KEY = 'image'
UPLOAD_LOCATION = 'files'
upload_parser = api.parser()
upload_parser.add_argument(UPLOAD_KEY,
                           location=UPLOAD_LOCATION,
                           type=FileStorage,
                           required=True)

@ns.route('/prediction')
class GanPrediction(Resource):
    @ns.doc(description='Predict the house number on the image using GAN model. ' +
            'Return 3 most probable digits with their probabilities',
            responses={
                200: "Success",
                400: "Bad request",
                500: "Internal server error"
                })
    @ns.expect(upload_parser)
    def post(self):
        try:
            image_file = request.files[UPLOAD_KEY]
            image = io.BytesIO(image_file.read())
            data = Image.open(image)
            data = data.resize((300,300))
            logging.info(type(data))
        except Exception as inst:
            return {'message': 'something wrong with incoming request. ' +
                               'Original message: {}'.format(inst)}, 400

        try:
            data = np.array(data) / 255.0
            data = np.expand_dims(data, 0)
            data = data.astype(np.float32)
            results = make_prediction(data)
            return {'prediction_result': results}, 200
        except Exception as inst:
            return {'message': 'internal error: {}'.format(inst)}, 500
