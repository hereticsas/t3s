"""
T3S configuration file.

Edit it to suit your needs:
    1. set the `${SERVER_NAME}` to the address and port of your API in the form:
    `{@server:port}` (e.g. `127.0.0.1:5000`)
    2. choose your server configuration: `default`, `dev`, `testing` or
    `production` and set it in the `configure_app()` function
    3. configure your TensorFlow models by setting the `${TF_MODELS}` dict:
        - 'dir' is the global directory containing all your models in separate
        subfolders
        - 'extractors' is a Python dict that contains specific features extractor
        for your models if necessary (see the Readme for more information)
"""

import os
import importlib

# FLASK SERVER CONFIGURATION
# ==========================
class BaseConfig(object):
    DEBUG = False
    TESTING = False

    # Base info
    SERVER_NAME = '127.0.0.1:5000'  # in the form: '@server:port'
    SITE_TITLE = 'Demo Server'

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
TF_MODELS = {
    'dir': '',
    'extractors': {}
}
