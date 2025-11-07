from typing import List, Tuple

import numpy as np
from paddleocr import PaddleOCR


def run_ocr_on_crops(img_bgr: np.ndarray, boxes: List[List], ocr_model: str) -> List[Tuple[List, str]]:
    # print("[run_ocr_on_crops]]")
    # print(boxes)

    H, W = img_bgr.shape[:2]
    out = []
    # for cluster in :
    for text_box in boxes:
        x, y, w, h = text_box
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        crop = img_bgr[y:y+h, x:x+w]

        if crop.size == 0:
            out.append((text_box, "[empty_crop_skipped]"))
            continue
        # NOTE: 모델 결정
        if ocr_model=="tesseract":
            text = ocr_pytesseract(crop)
        elif ocr_model=="paddle":
            text = ocr_paddle(crop)
        elif ocr_model=="easyocr":
            text = ocr_easy(crop)
        elif ocr_model=="pororo":
            text = ocr_pororo(crop)
        else:
            text = ""
        out.append((text_box, text if text else "[empty_text]"))
    return out

def ocr_pytesseract(img_bgr: np.ndarray) -> str:
    try:
        import pytesseract
        cfg = "--oem 3 --psm 4"
        return pytesseract.image_to_string(img_bgr, lang="kor", config=cfg).strip()
    except Exception as e:
        return f"[tesseract_unavailable: {e}]"


_paddle_ocr = None
def get_paddle_ocr() -> PaddleOCR:
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(lang='korean')
    return _paddle_ocr

def ocr_paddle(img_bgr: np.ndarray) -> str:
    try:
        ocr = get_paddle_ocr()
        result = ocr.ocr(img_bgr)
        texts = result[0].get("rec_texts")
        print(texts)
        return "".join(texts)
    except Exception as e:
        return f"[paddle_unavailable: {e}]"

def ocr_easy(img_bgr: np.ndarray) -> str:
    try:
        from easyocr import Reader
        reader = Reader(lang_list=['ko', 'en'], gpu=True)
        output = reader.readtext(img_bgr, paragraph="True", detail="False")
        return "".join(output)
    except Exception as e:
        return f"[easy_unavailable: {e}]"


def _init_pororo():
    import torch
    from ocr_service.pororo.brainocr import Reader
    from ocr_service.pororo.tasks.utils.download_utils import download_or_load

    LANG = "ko"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    det_model_path = download_or_load(f"misc/craft.pt",LANG)
    rec_model_path = download_or_load(f"misc/brainocr.pt",LANG)
    opt_fp = download_or_load(f"misc/ocr-opt.txt",LANG)

    reader = Reader(
        LANG,
        det_model_ckpt_fp=det_model_path,
        rec_model_ckpt_fp=rec_model_path,
        opt_fp=opt_fp,
        device=DEVICE,
    )
    reader.detector.to(DEVICE)
    reader.recognizer.to(DEVICE)
    return reader

def ocr_pororo(img_bgr: np.ndarray) -> str:
    try:
        from ocr_service.pororo.tasks.optical_character_recognition import PororoOCR
        from ocr_service.pororo.tasks.utils.download_utils import download_or_load
        from ocr_service.pororo.tasks.utils.base import TaskConfig

        brain_ocr = _init_pororo()
        results = brain_ocr(img_bgr)
        texts = []
        for result in results:
            (_, res, _) = result
            texts.append(res)
        return " ".join(texts)
    except Exception as e:
        return f"[easy_unavailable: {e}]"

# def ocr_deepseek(img_bgr: np.ndarray) -> str:
#     try:
#         from transformers import AutoModel, AutoTokenizer
#         import torch
#         import os
#
#         model_name = 'deepseek-ai/DeepSeek-OCR'
#
#         tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#         model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True,
#                                           use_safetensors=True)
#         model = model.eval().cuda().to(torch.bfloat16)
#
#         prompt = "<image>\n<|grounding|>Convert the document to markdown. "
#         image_file = 'your_image.jpg'
#
#         res = model.infer(tokenizer, prompt=prompt, image_file=image_file, base_size=1024,
#                       image_size=640, crop_mode=True, save_results=True, test_compress=True)
#         print(res)
#         return res
#     except Exception as e:
#         return f"[tesseract_unavailable: {e}]"

