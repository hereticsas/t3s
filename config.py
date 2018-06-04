"""
T3S configuration file.
"""

import os

# FLASK SERVER CONFIGURATION
# ==========================
class BaseConfig(object):
    DEBUG = False
    TESTING = False

    # Base info
    SERVER_NAME = '127.0.0.1:4000'  # in the form: '@server:port'

class DevelopmentConfig(BaseConfig):
    DEBUG = True
    TESTING = True


class TestingConfig(BaseConfig):
    DEBUG = False
    TESTING = True


class ProductionConfig(BaseConfig):
    DEBUG = False
    TESTING = False

config = {
    'dev': 'config.DevelopmentConfig',
    'test': 'config.TestingConfig',
    'prod': 'config.ProductionConfig',
    'default': 'config.DevelopmentConfig'
}

def configure_app(app):
    config_name = os.getenv('FLASK_CONFIGURATION', 'default')
    app.config.from_object(config[config_name])


# TENSORFLOW CONFIGURATION
# ========================
TF_MODEL_DIR = '/tmp/email_predictor_model/1528091811/'
