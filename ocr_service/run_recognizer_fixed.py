"""
Pororo BrainOCR 호환 간단 실행 스크립트
=====================================
기존 코드에서 최소한의 수정으로 Pororo 설정 적용
"""

import torch
from PIL import Image
from ocr_service.text_recognizer import TextRecognizer


class PororoOpt:
    def __init__(self):
        # === 파일 경로 및 디바이스 ===
        self.saved_model = "./weights/text_recognizer/TPS-ResNet-BiLSTM-CTC.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # === 배치 처리 설정 ===
        self.batch_size = 192  # 변경: 768 → 192 (Pororo 기본값)
        self.batch_max_length = 25  # 변경: 100 → 25 (한국어 한 줄 텍스트)

        # === 이미지 전처리 ===
        self.imgH = 64  # 유지 (적절한 높이)
        self.imgW = 600  # ⭐ 핵심 변경: 100 → 600 (한국어 텍스트에 최적)
        self.PAD = True  # 유지 (비율 유지 패딩)
        self.rgb = False  # 유지 (그레이스케일)

        # === 모델 아키텍처 (TRBA) ===
        self.Transformation = "TPS"  # 유지
        self.FeatureExtraction = "ResNet"  # 변경: VGG → ResNet (더 강력한 특징 추출)
        self.SequenceModeling = "BiLSTM"  # 유지
        self.Prediction = "CTC"  # 유지

        # === 네트워크 파라미터 ===
        self.num_fiducial = 2589  # 유지
        self.input_channel = 1  # 유지
        self.output_channel = 512  # 유지
        self.hidden_size = 256  # 변경: 512 → 256 (Pororo 기본값)


if __name__ == "__main__":
    # 1. 설정 생성
    opt = PororoOpt()

    # 2. 디바이스 설정
    device = torch.device(opt.device)
    opt.device = device

    print(f"🔧 설정 정보:")
    print(f"  - 디바이스: {opt.device}")
    print(f"  - 배치 크기: {opt.batch_size}")
    print(f"  - 이미지 크기: {opt.imgH}x{opt.imgW}")
    print(f"  - 모델: {opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}")
    print()

    # 3. TextRecognizer 초기화
    recognizer = TextRecognizer(opt, model_type="")

    print("✅ 모델 로드 완료")
    print()

    # 4. 테스트 이미지 준비
    # 주의: 각 이미지는 '한 줄'의 텍스트만 포함해야 함!
    img_paths = [
        "../resource/line_1.png",  # "정치인을 꿈꾼다면" 라인
        "../resource/line_2.png",  # "즐거움 같은 것이..." 라인
        # 필요시 더 추가
    ]

    # 5. 이미지 로드
    try:
        pil_images = [Image.open(p).convert("L") for p in img_paths]
        print(f"📁 {len(pil_images)}개 이미지 로드 완료")
    except FileNotFoundError as e:
        print(f"❌ 오류: 테스트 이미지를 찾을 수 없습니다.")
        print(f"   {e.filename}")
        print("\n💡 해결 방법:")
        print("1. 원본 이미지(image.png)를 한 줄씩 자르기")
        print("2. 각 라인을 line_1.png, line_2.png로 저장")
        print("3. ../resource/ 폴더에 저장")
        exit(1)

    # 6. OCR 실행
    print("\n🔍 OCR 시작...")
    result = recognizer.extract_text(pil_images)

    # 7. 결과 출력
    print("\n📋 OCR 결과:")
    print("-" * 50)
    if isinstance(result, list):
        for i, text in enumerate(result, 1):
            print(f"{i}. {text}")
    else:
        print(result)