from flask import Blueprint, request, jsonify, current_app
import os
from werkzeug.utils import secure_filename
from deepface import DeepFace
import pandas as pd
from ..service.face_recognition_service import FaceRecognitionService

deep_learning_controller = Blueprint('main', __name__, url_prefix='/api')

face_recognition_service = FaceRecognitionService()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 파일 확장자 허용 여부 확인 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@deep_learning_controller.route('/face-recognition', methods=['POST'])
def face_recognition():
    # 요청에서 파일이 포함되었는지 확인
    if 'file' not in request.files:
        return jsonify({"error": "요청에 파일이 없습니다"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "업로드할 파일이 없습니다"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # try:
        #     # 얼굴인식 로직 추가
        #
        # except Exception as e:
        #     return jsonify({"face recognition error": str(e)}), 500
        return jsonify({"file": file_path}), 200

    else:
        return jsonify({"error": "허용되지 않는 파일 형식입니다"}), 400