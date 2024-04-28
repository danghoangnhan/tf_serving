import logging.config

import settings
from flask import Flask, Blueprint
from api.restplus import api
import sys
sys.path.append('./.')
from tf_client.api.cnn.client import cnn_namespace
from tf_client.api.lstm.client import rhm_namespace

# create Flask application
application = Flask(__name__)

# # load logging confoguration and create log object
# logging.config.fileConfig('./logging.conf')
# log = logging.getLogger(__name__)


def __get_flask_server_params__(flaskConfig:settings.FlaskConfig):
    '''
    Returns connection parameters of the Flask application

    :return: Tripple of server name, server port and debug settings
    '''
    server_name = settings.get_env_var_setting('FLASK_SERVER_NAME', flaskConfig.DEFAULT_FLASK_SERVER_NAME)
    server_port = int(settings.get_env_var_setting('FLASK_SERVER_PORT', flaskConfig.DEFAULT_FLASK_SERVER_PORT))
    flask_debug = settings.get_env_var_setting('FLASK_DEBUG', flaskConfig.DEFAULT_FLASK_DEBUG)

    flask_debug = True if flask_debug == '1' else False

    return server_name, server_port, flask_debug



def configure_app(flask_app,restPlusConfig):
    '''
    Configure Flask application

    :param flask_app: instance of Flask() class
    '''
    flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = restPlusConfig.RESTPLUS_SWAGGER_UI_DOC_EXPANSION
    flask_app.config['RESTPLUS_VALIDATE'] = restPlusConfig.RESTPLUS_VALIDATE
    flask_app.config['RESTPLUS_MASK_SWAGGER'] = restPlusConfig.RESTPLUS_MASK_SWAGGER
    flask_app.config['ERROR_404_HELP'] = restPlusConfig.RESTPLUS_ERROR_404_HELP


def initialize_app(flask_app,restPlusConfig):
    '''
    Initialize Flask application with Flask-RestPlus
    :param flask_app: instance of Flask() class
    '''
    blueprint = Blueprint('tf_api', __name__, url_prefix='/tf_api')

    configure_app(flask_app,restPlusConfig)
    api.init_app(blueprint)
    
    api.add_namespace(cnn_namespace)
    api.add_namespace(rhm_namespace)

    flask_app.register_blueprint(blueprint)

if __name__ == '__main__':
    server_name, server_port, flask_debug = __get_flask_server_params__(settings.FlaskConfig)
    initialize_app(application,settings.RESTPLUSConfig)
    application.run(debug=flask_debug, host=server_name, port=server_port)
