from flask import Flask
from flask_restful import Api

import sys

import config
from t3s import T3S

app = Flask(__name__)
api = Api(app)


@app.route('/favicon.ico')
def favicon():
    return ''

api.add_resource(T3S, '/<string:data_input>')


if __name__ == '__main__':
    # Set app configuration
    config.configure_app(app)

    # Start server
    app.run()
