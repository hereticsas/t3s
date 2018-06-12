from flask import Flask, render_template
from flask_restful import Api

import os

import config
from t3s import T3S

app = Flask(__name__)
api = Api(app)


@app.route('/favicon.ico')
def favicon():
    return ''

@app.route('/')
def index():
    if config.TF_MODELS['dir'] == '':
        return render_template(
            'error.html',
            SITE_TITLE=app.config['SITE_TITLE'],
            error='You need to set the models directory.')

    models_list = [f for f in os.listdir(config.TF_MODELS['dir']) if not f[0] == '.']
    if len(models_list) == 0:
        return render_template(
            'error.html',
            SITE_TITLE=app.config['SITE_TITLE'],
            error='It seems the models directory is empty.')
    return render_template(
        'index.html',
        SITE_TITLE=app.config['SITE_TITLE'],
        SERVER_NAME=app.config['SERVER_NAME'],
        models_list=models_list)

@app.route('/<model>')
def modelpage(model=None):
    if model is None:
        return render_template(
            'error.html',
            SITE_TITLE=app.config['SITE_TITLE'],
            error='This model seems incorrect.')
    return render_template(
        'modelpage.html',
        SITE_TITLE=app.config['SITE_TITLE'],
        SERVER_NAME=app.config['SERVER_NAME'],
        model=model)

api.add_resource(T3S, '/<model>/<string:data_input>')


if __name__ == '__main__':
    # Set app configuration
    config.configure_app(app)

    # Start server
    app.run()
