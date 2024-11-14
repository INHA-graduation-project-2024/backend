import os

from flask import Flask
from .controller import register_blueprints

def create_app():
    app = Flask(__name__)

    register_blueprints(app)

    # 이미지가 저장될 폴더를 설정
    # app.config['UPLOAD_FOLDER'] = './static/face'
    # app.config['UPLOAD_PASSIVE_FOLDER'] = './static/passive'

    # 이미지가 저장될 폴더를 절대 경로로 설정
    app.config['UPLOAD_FOLDER'] = os.path.abspath('./static/face')
    app.config['UPLOAD_PASSIVE_FOLDER'] = os.path.abspath('./static/passive')

    # 폴더가 없는 경우 생성
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['UPLOAD_PASSIVE_FOLDER']):
        os.makedirs(app.config['UPLOAD_PASSIVE_FOLDER'])

    return app
