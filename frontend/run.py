
"""
Changes are made according to scotch tutorial
"""
# from flask import session
from app import create_app
# from flask.ext.session import Session
import os

config_name = os.getenv('FLASK_CONFIG')
app = create_app(config_name)

if __name__ == '__main__':
    # app.secret_key = 'abcdefgh123456788'
    # app.config['SESSION_TYPE'] = 'filesystem'
    # sess = Session()
    # sess.init_app(app)

    app.debug = True
    app.run(port=5000,debug=True)





