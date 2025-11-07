import torch
from PIL import Image

from ocr_service.text_recognizer import TextRecognizer


# 간단한 설정 객체 정의
class Opt:
    def __init__(self):
        self.saved_model = "./weights/text_recognizer.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 192
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.PAD = False
        self.rgb = False
        self.Transformation = "TPS"
        self.FeatureExtraction = "ResNet"
        self.SequenceModeling = "BiLSTM"
        self.Prediction = "CTC"
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256


if __name__ == "__main__":
    opt = Opt()

    device = torch.device("cuda" if (torch.cuda.is_available() and getattr(opt, "device", "cpu") != "cpu") else "cpu")
    opt.device = device

    recognizer = TextRecognizer(opt, model_type="")
    print(recognizer.model)  # 모델 구조 확인 (선택 사항)

    img_paths = [
        "/mnt/c/SSAFY/ocr_pjt/resource/image.png",
        # "/mnt/c/SSAFY/ocr_pjt/resource/강아지 어쩌구 웹툰/2.jpg",
    ]

    try:
        pil_images = [Image.open(p) for p in img_paths]
    except FileNotFoundError as e:
        print(f"오류: 테스트 이미지를 찾을 수 없습니다. {e.filename}")
        exit()

    result = recognizer.extract_text(pil_images)

    print("🔍 OCR 결과:")
    for line in result:
        print(line)