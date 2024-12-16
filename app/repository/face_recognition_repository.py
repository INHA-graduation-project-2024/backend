# import tensorflow as tf
from app.config.init_chroma import init_chroma
import uuid
from datetime import datetime

class FaceRecognitionRepository:
    def __init__(self):
        self.collection = init_chroma()
        # self.model_path = ''
        # self.model = None

    def add_face_data(self, embedding, name):
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")  # 현재 시간 (년월일시분초)
        random_uuid = uuid.uuid4().hex[:8]  # 랜덤 UUID의 앞 8자리 사용
        face_id = f"{name}_{current_time}_{random_uuid}"  # 이름, 시간, 랜덤 UUID 조합

        try:
            query_result = self.collection.add(
                embeddings=[embedding],
                metadatas=[{
                    'name': name,
                }],
                # ids=[face_id] #한명 당 얼굴 하나라...
                ids = [name]
            ) # add 성공 시 리턴이 None
            return
        except Exception as e:
            raise RuntimeError(f"Failed to add face data: {str(e)}")

    def search_face_data(self, embedding): # data말고 query 조건 넣을수도
        # 가장 가까운 항목 조회
        query_result = self.collection.query(
            query_embeddings=[embedding],
            n_results=1
        )
        return query_result

    def search_data_by_ids(self, data):
        pass

    # def load_model(self):
    #     if self.model is None:
    #         self.model = tf.keras.models.load_model(self.model_path)
    #     return self.model
