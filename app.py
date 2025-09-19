# app.py
import os
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template

# ---- TFLite runtime fallback ----
try:
    import tflite_runtime.interpreter as tflite  # سبک‌تر
except Exception:
    import tensorflow as tf
    tflite = tf.lite
# optional: gdown to auto-download model if missing
try:
    import gdown
except Exception:
    gdown = None

app = Flask(__name__)

MODEL_PATH = "model.tflite"
MODEL_GDRIVE_URL = "https://drive.google.com/uc?id=1-e7UBpGkYgQlrfrOfpP9TtJ7UZt8A2rx"

# ---- Flask app ----
app = Flask(__name__)


CLASS_LABELS = ["Bad", "Explode", "Good"]

# ---- Load TFLite model ----
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Put your tflite model at project root.")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ===================
# Utilities
# ===================
def required_image_size():
    """Detect image input size from TFLite model"""
    for d in input_details:
        shape = d.get("shape", [])
        if hasattr(shape, "__len__") and len(shape) == 4:
            return int(shape[1]), int(shape[2])
    return 256, 256

def apply_circle_black_background(gray_img, center=None, radius=None):
    """داخل دایره تصویر اصلی بمونه، بیرون دایره مشکی (0) بشه"""
    h, w = gray_img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(h, w) // 2   # نصف ضلع کوچک تصویر
    mask = np.zeros_like(gray_img, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    #black_bg = np.zeros_like(gray_img, dtype=np.uint8)
    masked = np.where(mask == 255, gray_img, 0)
    return masked


def apply_circle_mask(gray_img, radius=180):
    """
    Only keep inside circle, set outside to black (0).
    """
    h, w = gray_img.shape[:2]
    center = (w // 2, h // 2)
    mask = np.zeros_like(gray_img, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    masked = np.where(mask == 255, gray_img, 0)
    return masked


def remove_green(image):
    """
    Remove green overlay by replacing green pixels with black.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40])   # محدوده سبز
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    image[mask > 0] = (0, 0, 0)
    return image


def preprocess_image_bytes(img_bytes, target_size=(256, 256), circle_radius=180):
    #arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    #if img_bytes is None:
        #raise ValueError("cv2.imdecode failed")

    #arr = remove_green(arr)
    gray = cv2.cvtColor(img_bytes, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)

    h, w = thresh.shape[:2]
    radius = min(h, w) // 3
    center = (w // 2, h // 2)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    circle_only = np.where(mask == 255, thresh, 0)


    # --- اول ماسک سیاه بیرون دایره ---
    #masked = np.where(mask == 255, gray, 0)

    # --- فیلتر و آستانه ---
    #blur = cv2.GaussianBlur(masked, (3, 3), 0)
    #_, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)

    # --- دوباره ماسک برای اطمینان ---
    #final = np.where(mask == 255, thresh, 0)

    # resize
    resized = cv2.resize(circle_only, (256,256))

    # Debug save
    cv2.imwrite("debug_thresh.png", resized)

    # Normalize
    norm = resized.astype(np.float32) / 255.0
    norm = norm[:, :, np.newaxis]

    # Make preview
    _, buf = cv2.imencode(".png", resized.astype(np.uint8))
    b64 = base64.b64encode(buf).decode("utf-8")
    preview_dataurl = "data:image/png;base64," + b64

    return norm, preview_dataurl


# ===================
# Inference
# ===================
def run_inference_and_debug(numeric_vector, imgB_arr, imgF_arr):
    """
    numeric_vector: shape (8,)
    imgB_arr, imgF_arr: (H, W, 1), normalized
    """
    numeric_vector = numeric_vector.astype(np.float32).copy()

    # ---- normalize numbers (مثل قبل) ----
    numeric_vector[0] = (numeric_vector[0] - 35) / (95 - 35)
    numeric_vector[1] = (numeric_vector[1] - 200) / (1500 - 200)
    numeric_vector[2] = (numeric_vector[2]) / 15.0
    numeric_vector[3] = (numeric_vector[3] - 0.61) / (1.057 - 0.61)
    numeric_vector[4] = (numeric_vector[4] - 0.608) / (1.01 - 0.608)
    numeric_vector[5] = numeric_vector[5]
    numeric_vector[6] = (numeric_vector[6]) / 133.53
    numeric_vector[7] = (numeric_vector[7]) / 5009.43

    img_count = 0
    for d in input_details:
        idx = d["index"]
        shape = d.get("shape", [])
        if len(shape) == 2:  # numeric
            vec = numeric_vector.astype(d["dtype"])[None, :]
            interpreter.set_tensor(idx, vec)
        elif len(shape) == 4:  # image
            arr = imgB_arr if img_count == 0 else imgF_arr
            arr_to_set = arr.astype(np.float32)[None, :, :, :]
            interpreter.set_tensor(idx, arr_to_set)
            img_count += 1

    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])
    out = np.squeeze(out)

    cls = int(np.argmax(out))
    prob = float(out[cls])
    label = CLASS_LABELS[cls] if cls < len(CLASS_LABELS) else f"Class {cls}"

    return {"label": label, "prob": prob, "raw": out.tolist()}, numeric_vector


# ===================
# Routes
# ===================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    nums = data.get("numbers", [])
    if len(nums) != 8:
        return jsonify({"error": "Need 8 numeric features"}), 400
    numeric_vector = np.array([float(x) if x != "" else 0.0 for x in nums], dtype=np.float32)

    imageB_b64 = data.get("imageB")
    imageF_b64 = data.get("imageF")
    if not imageB_b64 or not imageF_b64:
        return jsonify({"error": "Both imageB and imageF required"}), 400

    imgB_bytes = base64.b64decode(imageB_b64.split(",")[1])
    imgF_bytes = base64.b64decode(imageF_b64.split(",")[1])

    H, W = required_image_size()
    arrB, previewB = preprocess_image_bytes(imgB_bytes, target_size=(H, W))
    arrF, previewF = preprocess_image_bytes(imgF_bytes, target_size=(H, W))

    result, normalized_vector = run_inference_and_debug(numeric_vector, arrB, arrF)

    return jsonify({
        "result": result,
        "previewB": previewB,
        "previewF": previewF,
        "normalized": normalized_vector.tolist(),
        "label": result["label"],
        "prob": result["prob"]
    })


@app.route("/healthz")
def healthz():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)