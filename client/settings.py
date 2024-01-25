# Flask settings
import os


DEFAULT_FLASK_SERVER_NAME = '0.0.0.0'
DEFAULT_FLASK_SERVER_PORT = '5001'
DEFAULT_FLASK_DEBUG = '1'  # Do not use debug mode in production

# Flask-Restplus settings
RESTPLUS_SWAGGER_UI_DOC_EXPANSION = 'list'
RESTPLUS_VALIDATE = True
RESTPLUS_MASK_SWAGGER = False
RESTPLUS_ERROR_404_HELP = False

# GAN client settings
DEFAULT_TF_SERVER_NAME = '0.0.0.0'
DEFAULT_TF_SERVER_PORT = 8501
GAN_MODEL_NAME = 'cnn'
GAN_MODEL_SIGNATURE_NAME = 'predict_images'
GAN_MODEL_INPUTS_KEY = 'conv2d_input'

def get_env_var_setting(env_var_name, default_value):
    '''
    Returns specified environment variable value. If it does not exist,
    returns a default value

    :param env_var_name: environment variable name
    :param default_value: default value to be returned if a variable does not exist
    :return: environment variable value
    '''
    try:
        env_var_value = os.environ[env_var_name]
    except:
        env_var_value = default_value

    return env_var_value