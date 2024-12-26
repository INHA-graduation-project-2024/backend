import cv2
from flask import Blueprint, request, jsonify, current_app
import os
from werkzeug.utils import secure_filename
import pandas as pd
from ..service.face_recognition_service import FaceRecognitionService
from ..service.passive_liveness_service import PassiveLivenessService
import subprocess

deep_learning_controller = Blueprint('main', __name__, url_prefix='/api')

face_recognition_service = FaceRecognitionService()
passive_liveness_service = PassiveLivenessService()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 추론 스크립트 절대 경로
INFERENCE_SCRIPT = "/home/nahyk64/root/mnt/inference.py"
# 추론 진행할 가상환경 절대 경로
INFERENCE_ENV_PYTHON = "/home/nahyk64/miniconda3/envs/final/bin/python"

# 파일 확장자 허용 여부 확인 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@deep_learning_controller.route('/join', methods=['POST'])
def add_user():
    # 요청에서 'name' 값이 있는지 확인
    name = request.form.get('name')
    print("Request.form:", request.form)  # name 필드 로그
    print("Request.files:", request.files)  # file 필드 로그

    if not name:
        return jsonify({"error": "name 필드가 없습니다"}), 400

    # 요청에서 파일이 포함되었는지 확인
    if 'file' not in request.files:
        return jsonify({"error": "요청에 파일이 없습니다"}), 400

    file = request.files['file']

    # 파일이 허용된 확장자인지 확인
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "허용되지 않은 파일 형식입니다"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            # 이미지를 face_recognition_service에 전달하여 처리
            face_recognition_service.add_user(file_path, name)

            return jsonify({"message": f"{name}의 얼굴 데이터가 성공적으로 추가되었습니다."}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "허용되지 않는 파일 형식입니다"}), 400



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

        try:
            test = face_recognition_service.face_recognition(file_path)
            return jsonify({"result": test})
            # 얼굴인식 로직 추가

        except Exception as e:
            return jsonify({"face recognition error": str(e)}), 500

    else:
        return jsonify({"error": "이미지 파일을 확인할 수 없습니다."}), 400

@deep_learning_controller.route('/passive', methods=['POST'])
def passive():
    # 이미지 전송 확인
    if 'file' not in request.files:
        return jsonify({'error': '요청에 이미지가 없습니다'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "업로드할 파일이 없습니다"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_PASSIVE_FOLDER'], filename)
        print('저장할 file-path', file_path)
        file.save(file_path)

        # TODO: 저장 후 제대로 저장이 안 된 경우를 고려해야 할까

        try:
            # 이미지 읽고 사이즈 조정
            image = cv2.imread(file_path)
            if image is None:
                return jsonify({"error": "이미지 파일을 읽을 수 없습니다."}), 400

            resized_image = cv2.resize(image, (640, 484))  # 이미지 크기 224x224로 조정

            # 다시 저장 (덮어쓰기)
            cv2.imwrite(file_path, resized_image)
            print("이미지 리사이징 완료:", file_path)

            faces, img = passive_liveness_service.face_detection(file_path)
            passive_result = passive_liveness_service.preprocessing(faces, img)
            return jsonify({"result": passive_result})

        except Exception as e:
            return jsonify({"passive liveness detection error": str(e)}), 500

    else:
        return jsonify({"error": "이미지 파일을 확인할 수 없습니다."}), 400
    

@deep_learning_controller.route('/active', methods=['POST'])
def active():
    # video와 audio 키 확인
    if 'audio' not in request.files:
        return jsonify({'error': '요청에 audio 파일이 없습니다'}), 400

    if 'video' not in request.files:
        return jsonify({'error': '요청에 video 파일이 없습니다'}), 400
    
    audio_file = request.files['audio']
    video_file = request.files['video']

    if not (audio_file.filename.endswith('.mp3') and video_file.filename.endswith('.mp4')):
        return jsonify({'error': '파일 확장자가 올바르지 않습니다. (audio: mp3, video: mp4)'}), 400
    
    # 파일 저장
    audio_path = os.path.join(current_app.config['UPLOAD_ACTIVE_FOLDER'], audio_file.filename)
    video_path = os.path.join(current_app.config['UPLOAD_ACTIVE_FOLDER'], video_file.filename)
    audio_file.save(audio_path)
    video_file.save(video_path)

    try:
        # 추론 스크립트 실행
        result = subprocess.run(
            [INFERENCE_ENV_PYTHON, INFERENCE_SCRIPT, '--audio', audio_path, '--video', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 결과 처리
        if result.returncode != 0:
            return jsonify({'error' : 'Inference failed', 'details': result.stderr.strip()}), 500
        
        # stdout에서 'Predicted Sentence' 부분 추출
        stdout_lines = result.stdout.splitlines()  # stdout을 줄 단위로 분리
        predicted_sentence = None
        for line in stdout_lines:
            if "Predicted Sentence" in line:  # 'Predicted Sentence'가 포함된 줄 찾기
                predicted_sentence = line.split("Predicted Sentence: ")[-1].strip()
                break

        if not predicted_sentence:
            return jsonify({
                'error': 'Prediction failed',
                'details': 'Predicted sentence not found in output.',
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip()
            }), 500
        
        # JSON 응답으로 반환
        return jsonify({'predicted_sentence': predicted_sentence})

    except Exception as e:
        return jsonify({'active liveness detection error': str(e)}), 500
    
    # finally:
    #     # 임시 파일 삭제
    #     if os.path.exists(audio_path):
    #         os.remove(audio_path)

    #     if os.path.exists(video_path):
    #         os.remove(video_path)
        
    #     npy_path = video_path.replace(".mp4", ".npy")
    #     if os.path.exists(npy_path):
    #         os.remove(npy_path)
