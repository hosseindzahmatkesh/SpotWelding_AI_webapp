import gdown
import os

MODEL_PATH = "model.tflite"
MODEL_URL = "https://drive.google.com/uc?id=1-e7UBpGkYgQlrfrOfpP9TtJ7UZt8A2rx"
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import base64, io, cv2

# Try to use lightweight tflite_runtime, fallback to tensorflow if not available
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    import tensorflow as tf
    tflite = tf.lite

app = Flask(__name__)

CLASS_LABELS = ["Bad", "Good", "Explode"]
MODEL_PATH = os.environ.get("MODEL_PATH", "model.tflite")

# --- Load TFLite model ---
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def required_image_size():
    for d in input_details:
        shape = d.get('shape', [])
        if hasattr(shape, "__len__") and len(shape) == 4:
            return int(shape[1]), int(shape[2])
    return 256, 256

def coerce_image_dtype(arr, dtype):
    if str(dtype) in ("float32", "float64"):
        if arr.max() > 1.0:  # if 0..255
            arr = arr / 255.0
        return arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = (arr * 255.0).round()
    return arr.astype(np.uint8)

def preprocess_image(image_bytes):
    """Convert uploaded image → grayscale → Gaussian blur → threshold → Canny → resize"""
    H, W = required_image_size()
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(img)

    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    _, arr = cv2.threshold(arr, 40, 255, cv2.THRESH_BINARY)
    arr = cv2.Canny(arr, 700, 1000)
    arr = cv2.resize(arr, (W, H))

    # دایره سبز (12mm = 45px تقریبی)
    center = (W // 2, H // 2)
    radius_px = 45
    cv2.circle(arr, center, radius_px, (255, 255, 255), 2)  # سفید برای نمای سیاه/سفید

    arr = arr[:, :, np.newaxis] / 255.0
    return arr

def run_inference(img1, img2, numeric_vector):
    for i, d in enumerate(input_details):
        shape = d.get('shape', [])
        idx = d['index']
        if hasattr(shape, "__len__") and len(shape) == 4:
            arr = img1 if i == 0 else img2
            arr = coerce_image_dtype(arr, d['dtype'])
            arr = arr[None, :, :, 0]
            interpreter.set_tensor(idx, arr)
        else:
            vec = numeric_vector.astype(d['dtype'])
            vec = vec[None, :]
            interpreter.set_tensor(idx, vec)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    out = np.squeeze(out)
    cls = int(np.argmax(out))
    prob = float(out[cls])
    label = CLASS_LABELS[cls] if cls < len(CLASS_LABELS) else f"Class {cls}"
    return {"label": label, "prob": prob, "raw": out.tolist()}

@app.route("/")
def index():
    h, w = required_image_size()
    return render_template("index.html", imgw=w, imgh=h, labels=CLASS_LABELS)

@app.route("/healthz")
def healthz():
    return "ok"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        image_data = data["image"].split(",")[1]
    except Exception:
        return jsonify({"error": "invalid image payload"}), 400

    img_bytes = base64.b64decode(image_data)
    arr = preprocess_image(img_bytes)

    nums = data.get("numbers", [])
    if not isinstance(nums, list) or len(nums) != 8:
        return jsonify({"error": "need 8 numeric features"}), 400
    try:
        numeric_vector = np.array([float(x) if x != "" else 0.0 for x in nums], dtype=np.float32)
    except Exception:
        return jsonify({"error": "non-numeric feature detected"}), 400

    result = run_inference(arr, arr, numeric_vector)

    # تصویر خروجی برای نمایش
    img_out = (arr.squeeze() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_out)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    result["processed_image"] = "data:image/png;base64," + img_b64
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
