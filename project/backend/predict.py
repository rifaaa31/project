import os
import json
from typing import Dict, Tuple

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    _TF_AVAILABLE = True
except Exception:
    tf = None
    load_model = None
    _TF_AVAILABLE = False

CONFIDENCE_THRESHOLD: float = 0.80
_CONFIG_LOADED = False


def _backend_dirs() -> Tuple[str, str]:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(backend_dir, "model")
    return backend_dir, model_dir


def _load_labels(model_dir: str) -> list:
    labels_path = os.path.join(model_dir, "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            return json.load(f)
    # Default order if labels file missing
    return ["nv", "mel", "bcc", "akiec", "bkl", "df", "vasc", "no_lesion"]


class ModelHolder:
    _model = None
    _labels = None
    _threshold = None
    _image_size = (64, 64)

    @classmethod
    def get_model_and_labels(cls):
        if cls._model is None:
            _, model_dir = _backend_dirs()
            model_path = os.path.join(model_dir, "skin_lesion_model.h5")
            if _TF_AVAILABLE and os.path.exists(model_path):
                cls._model = load_model(model_path)
            else:
                # Lightweight numpy-based fallback predictor; returns uniform probabilities
                class NumpyFallback:
                    def predict(self, x, verbose=0):
                        batch_size = x.shape[0]
                        probs = np.full((batch_size, 8), 1.0 / 8.0, dtype=np.float32)
                        return probs
                cls._model = NumpyFallback()
        if cls._labels is None:
            _, model_dir = _backend_dirs()
            cls._labels = _load_labels(model_dir)
        # Load threshold/config if available
        if cls._threshold is None:
            _, model_dir = _backend_dirs()
            config_path = os.path.join(model_dir, "config.json")
            thr = None
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        cfg = json.load(f)
                    thr = float(cfg.get("confidence_threshold", CONFIDENCE_THRESHOLD))
                    size = cfg.get("image_size")
                    if isinstance(size, (list, tuple)) and len(size) == 2:
                        cls._image_size = (int(size[0]), int(size[1]))
                except Exception:
                    thr = None
            cls._threshold = thr if thr is not None else CONFIDENCE_THRESHOLD
        return cls._model, cls._labels


def preprocess_image(image: Image.Image, target_size=(64, 64)) -> np.ndarray:
    image = image.convert("RGB").resize(target_size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def severity_for(label: str) -> str:
    high = {"mel", "akiec"}
    medium = {"bcc"}
    low = {"nv", "bkl", "df", "vasc"}
    if label == "no_lesion":
        return "NONE"
    if label in high:
        return "HIGH"
    if label in medium:
        return "MEDIUM"
    if label in low:
        return "LOW"
    return "LOW"


def predict_from_pil(image: Image.Image) -> Dict:
    model, labels = ModelHolder.get_model_and_labels()
    # Use configured image size if available
    target_size = ModelHolder._image_size if isinstance(ModelHolder._image_size, tuple) else (64, 64)
    arr = preprocess_image(image, target_size=target_size)
    probs = model.predict(arr, verbose=0)[0]
    top_index = int(np.argmax(probs))
    top_prob = float(probs[top_index])
    predicted_label = labels[top_index]

    threshold = ModelHolder._threshold if ModelHolder._threshold is not None else CONFIDENCE_THRESHOLD
    if top_prob < threshold:
        return {
            "class": "Wrong Image — Not a Skin Lesion",
            "confidence": round(top_prob, 4),
            "severity": "NONE",
            "is_wrong_image": True,
        }

    severity = severity_for(predicted_label)
    return {
        "class": predicted_label,
        "confidence": round(top_prob, 4),
        "severity": severity,
        "is_wrong_image": False,
    }


def predict_from_file(file_path: str) -> Dict:
    with Image.open(file_path) as img:
        return predict_from_pil(img)


def predict_from_ndarray_rgb(frame_bgr: np.ndarray) -> Dict:
    # Convert BGR (OpenCV) to RGB and predict
    rgb = frame_bgr[:, :, ::-1]
    img = Image.fromarray(rgb)
    return predict_from_pil(img)
