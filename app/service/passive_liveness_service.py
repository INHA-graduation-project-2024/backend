import os
import cv2 as cv
import torch
from torchvision import transforms
from ..ML_model.Model import DeePixBiS, MobileNet

# 현재 파일의 디렉토리 경로
base_dir = os.path.dirname(os.path.abspath(__file__))

# # 모델 로드
# model = DeePixBiS()
# model.load_state_dict(torch.load('../ML_model/DeePixBiS.pth'))
# model.eval()

# 모델 경로를 절대 경로로 설정
# model_path = os.path.join(base_dir, '../ML_model/DeePixBiS.pth')
model_path = os.path.join(base_dir, '../ML_model/MobilNet_epoch200_lr0.0001_noscheduler.pth')
haar_cascade_path = os.path.join(base_dir, '../ML_model/haarface.xml')

# 모델 로드
# model = DeePixBiS()
model = MobileNet()
# model.load_state_dict(torch.load(model_path))
# 모델 로드 시 map_location 옵션 추가
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.eval()

# 전처리..
tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

faceClassifier = cv.CascadeClassifier(haar_cascade_path)

# 이미지 읽기
# 회색조 변환 및 얼굴 검출
# 얼굴 검출 및 전처리
# 모델 예측
# 스푸핑 판별 결과 저장
# JSON 형태로 결과 반환

class PassiveLivenessService:
    def __init__(self):
        pass

    # @staticmethod
    def face_detection(self, img_path):
        img = cv.imread(img_path)
        if img is None:
            raise ValueError(f"이미지를 읽어오지 못했습니다: {img_path}")

        try:
            grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)
            return faces, img
        except Exception as e:
            raise RuntimeError(f"얼굴 검출 중 오류가 발생했습니다: {str(e)}")

    # @staticmethod
    def preprocessing(self, faces, img):
        results = []
        for x, y, w, h in faces:
            faceRegion = img[y:y + h, x:x + w]
            faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)

            faceRegion = tfms(faceRegion)
            faceRegion = faceRegion.unsqueeze(0)

            # 모델 예측
            with torch.no_grad():
                # mask, binary = model(faceRegion) #이건 densenet
                # print("mask", mask, "\nbinary", binary)

                mask = model.forward(faceRegion)
                print(mask)
                res = torch.mean(mask).item()

            # 스푸핑 판별 결과 저장
            result_text = 'Real' if res >= 0.5 else 'Fake'
            results.append({
                'face_position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'prediction': result_text,
                'confidence': res
            })

        # 스푸핑 판별 결과 반환
        return results