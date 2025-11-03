import sys
import os
import torch
import pprint

sys.path.append(os.path.abspath("."))

# --- 1. 핵심 모듈 임포트 ---
from pororo.tasks.optical_character_recognition import PororoOCR
from pororo import brainocr
from pororo.tasks.utils.download_utils import download_or_load
from pororo.tasks.utils.base import TaskConfig

print("커스텀 OCR 엔진 생성을 시작합니다...")

# --- 2. 설정 정의 ---
LANG = "ko"  # "ko" 또는 "en"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 언어: {LANG}, 사용 디바이스: {DEVICE}")

# --- 3. 모델 파일 다운로드 (Factory가 하던 일) ---
# 원본 파일의 PororoOcrFactory.load() 메소드 로직
print("필요한 모델 파일을 다운로드합니다... (처음 실행 시 시간이 걸림)")
det_model_path = download_or_load(
    f"misc/craft.pt",
    LANG,
)
rec_model_path = download_or_load(
    f"misc/brainocr.pt",
    LANG,
)
opt_fp = download_or_load(
    f"misc/ocr-opt.txt",
    LANG,
)
print("모델 다운로드 완료.")

# --- 4. 핵심 엔진(Reader) 로드 ---
# brainocr.Reader 객체를 생성합니다. (이것이 실제 모델)
core_model = brainocr.Reader(
    LANG,
    det_model_ckpt_fp=det_model_path,
    rec_model_ckpt_fp=rec_model_path,
    opt_fp=opt_fp,
    device=DEVICE,
)
core_model.detector.to(DEVICE)
core_model.recognizer.to(DEVICE)

# --- 5. 설정(config) 객체 생성 ---
# PororoOCR 래퍼는 model과 config를 인자로 받습니다.
# PororoOcrFactory가 하던 것처럼 간단한 Config 객체를 만듭니다.
# "Available tasks are ['mrc', 'rc', 'qa', 'question_answering', 'machine_reading_comprehension', 'reading_comprehension', 'sentiment', 'sentiment_analysis', 'nli', 'natural_language_inference', 'inference', 'fill', 'fill_in_blank', 'fib', 'para', 'pi', 'cse', 'contextual_subword_embedding', 'similarity', 'sts', 'semantic_textual_similarity', 'sentence_similarity', 'sentvec', 'sentence_embedding', 'sentence_vector', 'se', 'inflection', 'morphological_inflection', 'g2p', 'grapheme_to_phoneme', 'grapheme_to_phoneme_conversion', 'w2v', 'wordvec', 'word2vec', 'word_vector', 'word_embedding', 'tokenize', 'tokenise', 'tokenization', 'tokenisation', 'tok', 'segmentation', 'seg', 'mt', 'machine_translation', 'translation', 'pos', 'tag', 'pos_tagging', 'tagging', 'const', 'constituency', 'constituency_parsing', 'cp', 'pg', 'collocation', 'collocate', 'col', 'word_translation', 'wt', 'summarization', 'summarisation', 'text_summarization', 'text_summarisation', 'summary', 'gec', 'review', 'review_scoring', 'lemmatization', 'lemmatisation', 'lemma', 'ner', 'named_entity_recognition', 'entity_recognition', 'zero-topic', 'dp', 'dep_parse', 'caption', 'captioning', 'asr', 'speech_recognition', 'st', 'speech_translation', 'ocr', 'srl', 'semantic_role_labeling', 'p2g', 'aes', 'essay', 'qg', 'question_generation', 'age_suitability']"
config = TaskConfig(lang= LANG, n_model= "brainocr", task= "ocr")

# --- 6. 래퍼(Wrapper) 클래스에 엔진 삽입 ---
# 드디어 ocr_engine 객체(PororoOCR의 인스턴스)를 생성합니다.
# PororoOCR(model=핵심엔진, config=설정)
ocr_engine = PororoOCR(model=core_model, config=config)

print("OCR 엔진 생성 완료!")

# --- 7. 실행 테스트 ---
image_path = "../resource/image1.png"

if os.path.exists(image_path):
    print(f"\n--- 테스트 실행 ({image_path}) ---")
    # ocr_engine(image_path)는 내부적으로 ocr_engine.predict(image_path)를 호출합니다.
    results = ocr_engine(image_path) 
    pprint.pprint(results)

    print("\n--- 상세 테스트 실행 ---")
    results_detail = ocr_engine(image_path, detail=True)
    pprint.pprint(results_detail)
else:
    print(f"\n[경고] 테스트 이미지를 찾을 수 없습니다: {image_path}")
    print("image_path 변수에 실제 이미지 경로를 입력하고 다시 실행하세요.")