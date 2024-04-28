# Flask settings
import os

class FlaskConfig:
    DEFAULT_FLASK_SERVER_NAME = '0.0.0.0'
    DEFAULT_FLASK_SERVER_PORT = '5001'
    DEFAULT_FLASK_DEBUG = '1'  # Do not use debug mode in production

# Flask-Restplus settings
class RESTPLUSConfig:
    RESTPLUS_SWAGGER_UI_DOC_EXPANSION = 'list'
    RESTPLUS_VALIDATE = True
    RESTPLUS_MASK_SWAGGER = False
    RESTPLUS_ERROR_404_HELP = False

class TFServerconfig:
    DEFAULT_TF_SERVER_NAME = '0.0.0.0'
    DEFAULT_TF_SERVER_PORT = 8501

# CNN client settings
class ModelConfig:
    def __init__(self,MODEL_NAME,MODEL_SIGNATURE_NAME,MODEL_INPUTS_KEY):
        self.MODEL_NAME = MODEL_NAME
        self.MODEL_SIGNATURE_NAME = MODEL_SIGNATURE_NAME
        self.MODEL_INPUTS_KEY = MODEL_INPUTS_KEY

    def log_config(self):
        """Logs all configuration details of the model."""
        config_info = (
            f"Model Name: {self.MODEL_NAME}, "
            f"Signature Name: {self.MODEL_SIGNATURE_NAME}, "
            f"Inputs Key: {self.MODEL_INPUTS_KEY}"
        )
        self.logger.info(config_info)

CNNModelConfig = ModelConfig(
    MODEL_NAME = 'cnn',
    MODEL_SIGNATURE_NAME = 'predict_images',
    MODEL_INPUTS_KEY = 'conv2d_input'
)
  

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