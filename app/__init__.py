from flask import Flask
from .controller import register_blueprints

def create_app():
    app = Flask(__name__)

    register_blueprints(app)

    # 이미지가 저장될 폴더를 설정
    app.config['UPLOAD_FOLDER'] = './static/face'

    return app
