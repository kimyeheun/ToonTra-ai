from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch

torch.manual_seed(42)
np.random.seed(42)

@dataclass
class FontPred:
    filename: str          # e.g., "CoveredByYourGrace.ttf"
    family: str            # e.g., "Covered By Your Grace"
    score: float           # softmax prob
    google_url: str        # fonts.google.com?query=...

class StoriaFontClassifier:
    def __init__(
        self,
        repo_id: str = "storia/font-classify-onnx",
        model_filename: str = "model.onnx",
        class_names_url: str = "https://raw.githubusercontent.com/Storia-AI/font-classify/main/class_names.txt",
        mapping_tsv_url: str = "https://raw.githubusercontent.com/Storia-AI/font-classify/main/google_fonts_mapping.tsv",
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        topk: int = 3,
    ):
        self.topk = topk
        self.device = device
        self._ready = False
        self._err: Optional[str] = None

        try:
            from huggingface_hub import hf_hub_download
            import onnxruntime as ort

            cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "storia_font")
            os.makedirs(cache_dir, exist_ok=True)

            # 1) Download ONNX
            onnx_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir=cache_dir)

            # 2) Download class names and mapping
            import requests
            cn_resp = requests.get(class_names_url, timeout=20)
            cn_resp.raise_for_status()
            self.class_names: List[str] = [ln.strip() for ln in cn_resp.text.splitlines() if ln.strip()]

            map_resp = requests.get(mapping_tsv_url, timeout=20)
            map_resp.raise_for_status()
            df_map = pd.read_csv(io.StringIO(map_resp.text), sep="\t")
            # Expect columns: filename, family (per README note). :contentReference[oaicite:4]{index=4}
            self.fn2family = dict(zip(df_map["filename"], df_map["family"]))

            # 3) Create ONNX session
            providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.sess = ort.InferenceSession(onnx_path, providers=providers)

            # 4) Input / output names (assume standard)
            self.input_name = self.sess.get_inputs()[0].name
            self.output_name = self.sess.get_outputs()[0].name



            self._ready = True
        except Exception as e:
            self._ready = False
            self._err = f"{type(e).__name__}: {e}"

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> Optional[str]:
        return self._err

    # --- Pre/Post ---
    @staticmethod
    def _prepare(crop_bgr: np.ndarray, size: int = 224) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Empty crop")
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        # center-pad to square, then resize
        h, w = img.shape[:2]
        side = max(h, w)
        pad = np.full((side, side, 3), 255, dtype=np.uint8)
        y0 = (side - h) // 2
        x0 = (side - w) // 2
        pad[y0:y0+h, x0:x0+w] = img
        img = cv2.resize(pad, (size, size), interpolation=cv2.INTER_AREA)
        # normalize to 0..1 and NCHW float32
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return x

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=1, keepdims=True)
        ex = np.exp(x)
        return ex / ex.sum(axis=1, keepdims=True)

    def predict_topk(self, crop_bgr: np.ndarray, topk: Optional[int] = None) -> List[FontPred]:
        """
        Return top-k predictions for a crop.
        If model unavailable, returns a deterministic stub based on mean intensity.
        """
        k = topk or self.topk

        # Fallback if model failed to init (no internet, etc.)
        if not self._ready:
            mean_val = float(crop_bgr.mean()) if crop_bgr.size else 0.0
            idx = int(mean_val) % 997
            filename = self.class_names[idx % len(self.class_names)] if hasattr(self, "class_names") else f"stub_{idx}.ttf"
            family = self.fn2family.get(filename, filename.replace(".ttf", "").replace("_", " ")) if hasattr(self, "fn2family") else filename
            url = f"https://fonts.google.com?query={family.replace(' ', '+')}"
            return [FontPred(filename, family, 1.0, url)]

        x = self._prepare(crop_bgr)
        logits = self.sess.run([self.output_name], {self.input_name: x})[0]  # (1, C)
        probs = self._softmax(logits).reshape(-1)
        top_idx = probs.argsort()[-k:][::-1]

        out: List[FontPred] = []
        for i in top_idx:
            filename = self.class_names[i]
            family = self.fn2family.get(filename, filename.replace(".ttf", "").replace("_", " "))
            url = f"https://fonts.google.com?query={family.replace(' ', '+')}"
            out.append(FontPred(filename=filename, family=family, score=float(probs[i]), google_url=url))
        return out
