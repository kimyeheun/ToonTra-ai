import logging.config
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_CONF_PATH = os.path.join(BASE_DIR, 'logging.conf')

# 설정 파일 로드
try:
    logging.config.fileConfig(LOG_CONF_PATH)
except FileNotFoundError:
    print(f"경고: 로깅 설정 파일을 찾을 수 없습니다. (경로: {LOG_CONF_PATH})")
    logging.basicConfig(level=logging.INFO)
except Exception as e:
    print(f"경고: 로깅 설정 중 오류 발생: {e}")
    logging.basicConfig(level=logging.INFO)

# 로거 생성 및 사용
logger = logging.getLogger(__name__)
logger.info("OCR_PJT 시작")
