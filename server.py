import io

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils.ocr_util import pil_to_bgr


class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int

app = FastAPI()
app.mount("/app/static", StaticFiles(directory="/home/yeheun/PycharmProjects/Cartoon_editor/app/static"), name="static")

@app.post("/api/ocr")
async def ocr_endpoint(img: UploadFile = File(...), boxes_json: str = Form(...)):
    import json
    boxes = [Box(**b) for b in json.loads(boxes_json)]
    data = await img.read()
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    bgr = pil_to_bgr(pil)
    H, W = bgr.shape[:2]

    out = []
    for b in boxes:
        x = max(0, min(b.x, W-1))
        y = max(0, min(b.y, H-1))
        w = max(1, min(b.w, W-x))
        h = max(1, min(b.h, H-y))
        crop = bgr[y:y+h, x:x+w]
        text = f"[stub_{int(crop.mean())%997}]"  # replace with real OCR
        out.append({"box": b.dict(), "text": text})
    return JSONResponse({"results": out})
