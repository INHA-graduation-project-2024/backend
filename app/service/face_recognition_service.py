# service/face_recognition_service.py
from ..repository.face_recognition_repository import FaceRecognitionRepository
from deepface import DeepFace
from PIL import Image
import numpy as np

class FaceRecognitionService:
    def __init__(self):
        self.model_repository = FaceRecognitionRepository()
        # self.model = self.model_repository.load_model()

    def feature_extraction(self, img_path): # facenet 모델을 사용한 feature 추출
        embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", detector_backend="mtcnn")[0]['embedding']
        # print(embedding)
        return embedding

    def add_user(self, img_path, name):
        try:
            processed_img_path = self.preprocess_image(img_path)

            embedding = self.feature_extraction(processed_img_path)

            self.model_repository.add_face_data(embedding, name)
        except Exception as e:
            raise RuntimeError(f"사용자 추가에 실패했습니다: {str(e)}")

    def face_recognition(self, img_path):
        try:
            # 이미지 전처리 시도, 실패하면 원본 이미지 사용
            processed_img_path = self.preprocess_image(img_path)

            # 얼굴 임베딩 추출
            embedding = self.feature_extraction(processed_img_path)

            # 유사한 얼굴을 검색
            result = self.model_repository.search_face_data(embedding)
            print('service, face_recognition 결과 : ', result)
            return result
        except Exception as e:
            raise RuntimeError(f"얼굴 인식에 실패했습니다: {str(e)}")

    def preprocess_image(self, img_path):
        # 이미지 전처리 로직
        try:
            # 얼굴을 정렬 (얼라인)
            faces = DeepFace.extract_faces(img_path, detector_backend="mtcnn", align=True)

            if not faces:
                raise RuntimeError("얼굴을 검출하지 못했습니다.")

            # 첫 번째 얼굴의 이미지 배열을 사용 (여러 얼굴 중 하나만 선택할 경우)
            face_data = faces[0]  # 첫 번째 얼굴만 사용 (여러 얼굴이 있을 경우 이 부분을 조정)
            face_image = face_data['face']  # Numpy 배열로 얼굴 이미지 추출

            # Numpy 배열을 이미지로 변환하여 기존 경로에 저장 (덮어쓰기)
            aligned_image = Image.fromarray((face_image * 255).astype(np.uint8))  # Numpy 배열을 이미지로 변환
            aligned_image.save(img_path)  # 원본 이미지 경로에 덮어쓰기

            return img_path  # 정렬된 이미지 경로 반환
        except Exception as e:
            print(f"이미지 전처리에 실패했습니다: {str(e)}. 원본 이미지를 사용합니다.")
            return img_path  # 전처리 실패 시 원본 이미지를 그대로 사용

    def postprocess_predictions(self, predictions):
        # 예측 후처리 로직
        pass