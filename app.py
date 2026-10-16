"""
ASD Model Tester - local Flask app for testing several .keras binary classifiers.

Class mapping:  0 = ASD (Autism Spectrum Disorder)
                1 = Non-ASD

Put your models in the ./models folder (or upload them from the web page).
Everything runs locally; images are processed in memory and never stored.

Environment variables (all optional):
    MODELS_DIR   folder with .keras files            (default: ./models)
    PREPROCESS   default preprocessing               (default: rescale)
                   rescale -> pixels / 255          -> [0, 1]
                   tf      -> (pixels / 127.5) - 1  -> [-1, 1]
                              (same as resnet_v2 / mobilenet_v2 preprocess_input)
                   none    -> raw 0-255 (model has its own preprocessing layers)
    RESIZE       bilinear | nearest | bicubic        (default: bilinear)
    THRESHOLD    decision threshold for class 1      (default: 0.5)
    MAX_LOADED   models kept in memory at once       (default: 2)
    PORT         server port                         (default: 5000)

Per-model overrides go in models/models.json, see README.md.
"""
import gc
import io
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", BASE_DIR / "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = MODELS_DIR / "models.json"

DEFAULTS = {
    "preprocess": os.environ.get("PREPROCESS", "rescale").lower(),
    "resize": os.environ.get("RESIZE", "bilinear").lower(),
    "threshold": float(os.environ.get("THRESHOLD", "0.5")),
}
MAX_LOADED = max(1, int(os.environ.get("MAX_LOADED", "2")))
PORT = int(os.environ.get("PORT", "5000"))

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGES = 50
PREPROCESS_MODES = {"rescale", "tf", "none"}
RESAMPLE = {"bilinear": Image.BILINEAR, "nearest": Image.NEAREST, "bicubic": Image.BICUBIC}
CLASS_NAMES = {0: "ASD", 1: "Non-ASD"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # room for model uploads

_cache: "OrderedDict[str, tuple]" = OrderedDict()  # name -> (model, (w, h))
_lock = threading.Lock()


# ---------------------------------------------------------------- models ----
def list_model_files():
    return sorted(p.name for p in MODELS_DIR.glob("*.keras") if p.is_file())


def load_config():
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def settings_for(name):
    s = dict(DEFAULTS)
    s.update({k: v for k, v in load_config().get(name, {}).items() if k in DEFAULTS})
    if s["preprocess"] not in PREPROCESS_MODES:
        s["preprocess"] = DEFAULTS["preprocess"]
    s["threshold"] = float(s["threshold"])
    return s


def get_model(name):
    """Load a model (cached). Only MAX_LOADED models stay in memory."""
    if name in _cache:
        _cache.move_to_end(name)
        return _cache[name]

    import tensorflow as tf

    path = MODELS_DIR / name
    try:
        model = tf.keras.models.load_model(path, compile=False)
    except Exception:
        # Needed if the model contains Lambda layers
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)

    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    size = (224, 224)
    if shape and len(shape) == 4 and shape[1] and shape[2]:
        size = (int(shape[2]), int(shape[1]))  # PIL wants (w, h)

    _cache[name] = (model, size)
    while len(_cache) > MAX_LOADED:
        _cache.popitem(last=False)
        gc.collect()
        tf.keras.backend.clear_session()
    return _cache[name]


def to_prob_non_asd(out, n):
    """Model output -> P(class 1 = Non-ASD) for every image. Handles sigmoid(1) or softmax(2)."""
    if isinstance(out, (list, tuple)):
        out = out[0]
    out = np.asarray(out, dtype=np.float64).reshape(n, -1)
    if out.shape[1] == 1:
        p = out[:, 0]
        return np.where((p < 0) | (p > 1), 1 / (1 + np.exp(-p)), p)  # logits -> sigmoid
    if out.shape[1] == 2:
        if not np.allclose(out.sum(axis=1), 1.0, atol=1e-3):  # logits -> softmax
            e = np.exp(out - out.max(axis=1, keepdims=True))
            out = e / e.sum(axis=1, keepdims=True)
        return out[:, 1]
    raise ValueError(f"Unexpected model output shape: {out.shape}")


def preprocess(img, size, resize, mode):
    arr = np.asarray(img.convert("RGB").resize(size, RESAMPLE.get(resize, Image.BILINEAR)), dtype=np.float32)
    if mode == "rescale":
        arr = arr / 255.0
    elif mode == "tf":
        arr = arr / 127.5 - 1.0
    return arr


def run_model(name, images, override):
    """Run one model on a list of PIL images. Returns (list of result dicts, ms)."""
    s = settings_for(name)
    mode = override if override in PREPROCESS_MODES else s["preprocess"]
    t0 = time.perf_counter()
    model, size = get_model(name)
    batch = np.stack([preprocess(im, size, s["resize"], mode) for im in images])
    x = [batch] * len(model.inputs) if len(model.inputs) > 1 else batch  # fusion models with 2 inputs
    probs = to_prob_non_asd(model.predict(x, batch_size=16, verbose=0), len(images))
    ms = (time.perf_counter() - t0) * 1000

    results = []
    for p in probs:
        p = float(p)
        label = 1 if p >= s["threshold"] else 0
        results.append({
            "label": label,
            "class_name": CLASS_NAMES[label],
            "confidence": (p if label == 1 else 1 - p) * 100,
            "prob_asd": (1 - p) * 100,
            "prob_non_asd": p * 100,
            "raw_output": p,
            "threshold": s["threshold"],
            "preprocess": mode,
        })
    return results, ms


# ---------------------------------------------------------------- routes ----
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def api_models():
    out = []
    for name in list_model_files():
        s = settings_for(name)
        out.append({
            "name": name,
            "size_mb": round((MODELS_DIR / name).stat().st_size / 1_048_576, 1),
            "loaded": name in _cache,
            **s,
        })
    return jsonify(models=out, folder=str(MODELS_DIR), max_loaded=MAX_LOADED)


@app.route("/api/models/upload", methods=["POST"])
def api_upload_model():
    f = request.files.get("model")
    if f is None or not f.filename:
        return jsonify(error="No model file uploaded."), 400
    name = secure_filename(f.filename)
    if not name.lower().endswith(".keras"):
        return jsonify(error="Only .keras files are supported."), 400
    dest = MODELS_DIR / name
    if dest.exists():
        return jsonify(error=f"'{name}' already exists in the models folder. Rename the file or delete the old one."), 409
    f.save(dest)
    return jsonify(name=name)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    files = request.files.getlist("images")
    model_names = request.form.getlist("models")
    allow_png = request.form.get("allow_png", "false").lower() == "true"
    override = request.form.get("preprocess", "default").lower()

    if not files or all(not f.filename for f in files):
        return jsonify(error="Add at least one image."), 400
    if len(files) > MAX_IMAGES:
        return jsonify(error=f"Too many images (max {MAX_IMAGES} per run)."), 400
    available = set(list_model_files())
    model_names = [m for m in dict.fromkeys(model_names)]  # unique, keep order
    if not model_names:
        return jsonify(error="Select at least one model."), 400
    unknown = [m for m in model_names if m not in available]
    if unknown:
        return jsonify(error=f"Model not found: {', '.join(unknown)}. Refresh the model list."), 404

    allowed_ext = {".jpg", ".jpeg"} | ({".png"} if allow_png else set())
    allowed_fmt = {"JPEG"} | ({"PNG"} if allow_png else set())

    # 1) validate every image
    entries = []
    for f in files:
        entry = {"filename": f.filename, "predictions": {}}
        ext = os.path.splitext(f.filename)[1].lower()
        data = f.read()
        if ext not in allowed_ext:
            entry["error"] = f"Unsupported file type '{ext}'. Use JPG/JPEG" + (" or PNG." if allow_png else " (enable PNG to allow .png).")
        elif len(data) > MAX_IMAGE_BYTES:
            entry["error"] = "File is larger than 10 MB."
        else:
            try:
                img = Image.open(io.BytesIO(data))
                fmt = (img.format or "").upper()
                if fmt not in allowed_fmt:
                    entry["error"] = f"File content is {fmt or 'unknown'}, which is not allowed."
                else:
                    img.load()
                    entry["_img"] = img
            except (UnidentifiedImageError, OSError):
                entry["error"] = "The file is not a valid image."
        entries.append(entry)

    valid = [e for e in entries if "_img" in e]

    # 2) run each selected model once on all valid images
    timings = {}
    if valid:
        with _lock:
            for name in model_names:
                try:
                    results, ms = run_model(name, [e["_img"] for e in valid], override)
                    timings[name] = ms
                    for e, r in zip(valid, results):
                        e["predictions"][name] = r
                except Exception as ex:  # show model problems in the UI
                    for e in valid:
                        e["predictions"][name] = {"error": f"{type(ex).__name__}: {ex}"}

    for e in entries:
        e.pop("_img", None)
    return jsonify(images=entries, models=model_names, timings_ms=timings)


if __name__ == "__main__":
    print(f"Models folder: {MODELS_DIR}  ({len(list_model_files())} .keras file(s) found)")
    # 127.0.0.1 = reachable from this machine only (local inference only)
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
