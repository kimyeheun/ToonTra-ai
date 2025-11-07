# 기본 파라미터 (일반 텍스트용)
basic = {
    'canvas_size': 2560,
    'mag_ratio': 1.0,
    'slope_ths': 0.1,
    'ycenter_ths': 0.5,
    'height_ths': 0.5,
    'width_ths': 0.5,
    'add_margin': 0.1,
    'min_size': 20,
    'text_threshold': 0.7,
    'low_text': 0.4,
    'link_threshold': 0.4,
}

# 효과음 전용 파라미터 (기존)
sfx_original = {
    'text_threshold': 0.5,
    'low_text': 0.2,
    'link_threshold': 0.05,
    'canvas_size': 1280,
    'mag_ratio': 1.0,
    'slope_ths': 0.1,
    'ycenter_ths': 1.0,
    'height_ths': 1.0,
    'width_ths': 0.5,
    'add_margin': 0.1,
    'min_size': 50,
}

# 효과음 최적화 파라미터 - 레벨 1 (약한 연결)
sfx_level1 = {
    'text_threshold': 0.4,  # 텍스트 검출 임계값을 낮춤 (더 많은 텍스트 검출)
    'low_text': 0.15,  # 낮은 신뢰도 텍스트도 포함
    'link_threshold': 0.02,  # 매우 낮은 링크 임계값 (글자 간 연결 강화)

    'canvas_size': 1280,
    'mag_ratio': 1.5,  # 이미지 확대로 세밀한 검출
    'slope_ths': 0.3,  # 기울기 허용치 증가 (방사형 텍스트 대응)

    'ycenter_ths': 1.5,  # Y축 중심 차이 허용치 증가
    'height_ths': 1.5,  # 높이 차이 허용치 증가
    'width_ths': 1.0,  # 너비 차이 허용치 증가
    'add_margin': 0.2,  # 마진 증가로 박스 확장
    'min_size': 30,  # 최소 크기 감소
}

# 효과음 최적화 파라미터 - 레벨 2 (강한 연결)
sfx_level2 = {
    'text_threshold': 0.35,  # 더 공격적인 텍스트 검출
    'low_text': 0.1,  # 매우 낮은 신뢰도도 포함
    'link_threshold': 0.01,  # 극도로 낮은 링크 임계값

    'canvas_size': 2048,  # 고해상도 처리
    'mag_ratio': 2.0,  # 2배 확대
    'slope_ths': 0.5,  # 큰 기울기도 허용

    'ycenter_ths': 2.0,  # 매우 큰 Y축 차이도 허용
    'height_ths': 2.0,  # 매우 큰 높이 차이도 허용
    'width_ths': 1.5,  # 큰 너비 차이도 허용
    'add_margin': 0.3,  # 큰 마진
    'min_size': 20,  # 작은 글자도 포함
}

# 방사형 효과음 전용 파라미터
sfx_radial = {
    'text_threshold': 0.45,
    'low_text': 0.2,
    'link_threshold': 0.03,  # 방사형 패턴을 위한 중간 수준

    'canvas_size': 1920,
    'mag_ratio': 1.8,
    'slope_ths': 0.7,  # 방사형 텍스트는 기울기가 다양함

    'ycenter_ths': 2.5,  # 방사형은 Y축 차이가 큼
    'height_ths': 2.5,  # 원근감으로 인한 크기 차이 허용
    'width_ths': 2.0,
    'add_margin': 0.25,
    'min_size': 25,
}

# 파라미터 설명
"""
주요 파라미터 설명:

1. text_threshold (0.0~1.0):
   - 낮을수록: 더 많은 영역을 텍스트로 검출
   - 높을수록: 확실한 텍스트만 검출
   - 효과음용: 0.35~0.5 권장

2. low_text (0.0~1.0):
   - 낮을수록: 희미한 텍스트도 검출
   - 높을수록: 선명한 텍스트만 검출
   - 효과음용: 0.1~0.2 권장

3. link_threshold (0.0~1.0):
   - 낮을수록: 멀리 떨어진 글자도 연결
   - 높을수록: 가까운 글자만 연결
   - 효과음용: 0.01~0.05 권장 (핵심 파라미터!)

4. slope_ths (0.0~1.0):
   - 기울기 허용 범위
   - 방사형 텍스트: 0.5 이상 권장

5. ycenter_ths, height_ths, width_ths:
   - 박스 병합 시 차이 허용치
   - 높을수록 다른 크기/위치의 박스도 병합
   - 효과음용: 1.5~2.5 권장

6. canvas_size:
   - 처리 해상도
   - 높을수록 정확하지만 느림

7. mag_ratio:
   - 이미지 확대 비율
   - 1.5~2.0 권장

사용 권장사항:
- 일반적인 효과음: sfx_level1
- 연결이 약한 효과음: sfx_level2
- 방사형 효과음: sfx_radial
"""