import os

from flask import Flask
from .controller import register_blueprints
from flask_cors import CORS

def create_app():
    app = Flask(__name__)

    register_blueprints(app)

    # CORS 설정: localhost:5173에서만 요청 허용 + 자격 증명 지원 (임시)
    CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}}, supports_credentials=True)

    # 이미지가 저장될 폴더를 설정
    # app.config['UPLOAD_FOLDER'] = './static/face'
    # app.config['UPLOAD_PASSIVE_FOLDER'] = './static/passive'

    # 이미지가 저장될 폴더를 절대 경로로 설정
    app.config['UPLOAD_FOLDER'] = os.path.abspath('./static/face')
    app.config['UPLOAD_PASSIVE_FOLDER'] = os.path.abspath('./static/passive')
    app.config['UPLOAD_ACTIVE_FOLDER'] = os.path.abspath('./static/active')

    # 폴더가 없는 경우 생성
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['UPLOAD_PASSIVE_FOLDER']):
        os.makedirs(app.config['UPLOAD_PASSIVE_FOLDER'])

    return app
