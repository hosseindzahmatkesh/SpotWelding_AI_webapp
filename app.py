# app.py
import os
import io
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
# tflite runtime fallback
try:
    import tflite_runtime.interpreter as tflite # type: ignore
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

# download model if missing (optional)
if not os.path.exists(MODEL_PATH) and gdown is not None:
    print("Downloading model...")
    gdown.download(MODEL_GDRIVE_URL, MODEL_PATH, quiet=False)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Put your tflite model at project root or enable gdown.")

# load interpreter
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_LABELS = ["Bad", "Explode", "Good"]


# -------------------
# Utilities / preprocessing
# -------------------
def required_image_size():
    # find first 4D input shape to get target HW (height,width)
    for d in input_details:
        shape = d.get("shape", [])
        if hasattr(shape, "__len__") and len(shape) == 4:
            return int(shape[1]), int(shape[2])
    return 256, 256

def resize_with_padding(img, target_size=(256, 256)):
    """Resize while keeping aspect ratio and padding with gray"""
    h, w = img.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    # create gray canvas
    canvas = np.full((th, tw), 128, dtype=img.dtype)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def apply_circle_gray_background(gray_img, center=None, radius=None):
    """Make outside of circle gray (128) and keep inside real gray image."""
    h, w = gray_img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = 110
    mask = np.zeros_like(gray_img, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    gray_bg = np.full_like(gray_img, 128)
    masked = np.where(mask == 255, gray_img, gray_bg)
    return masked

def remove_green(image):
    """
    حذف رنگ سبز (دایره overlay) با ماسک HSV
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40])   # محدوده سبز روشن
    upper = np.array([85, 255, 255]) # محدوده سبز تیره
    mask = cv2.inRange(hsv, lower, upper)
    # جایگزینی پیکسل‌های سبز با سفید
    image[mask > 0] = (0, 0, 0)
    return image

def preprocess_image_bytes(img_bytes, target_size=(256, 256), circle_radius=None):
    """
    Decode base64 bytes -> حذف سبز -> RGB -> gray -> apply circle-gray-bg -> blur -> threshold
    -> resize -> normalize (0..1) -> add channel
    Returns: (normalized_array, preview_base64_png)
    """
    # decode bytes -> BGR array using cv2.imdecode
    try:
        arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as e:
        raise RuntimeError(f"Failed decode image bytes: {e}")

    # حذف سبز قبل از پردازش
    arr = remove_green(arr)

    # convert to gray
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    if circle_radius is None:
       # radius = 180
    else:
        radius = circle_radius

    #center = (w // 2, h // 2)
    #masked = apply_circle_gray_background(gray, center=center, radius=radius)

    # blur + threshold
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _,thresh = cv2.threshold(
        blur, 40, 255,
        cv2.THRESH_BINARY,
        )

    # resize with padding (حفظ نسبت تصویر)
    processed = cv2.resize(thresh,(256,256))

    # --- Debug: ذخیره تصویر threshold برای بررسی ---
    cv2.imwrite("debug_thresh.png", processed)

    # normalized (0..1) for model
    norm = processed / 255.0
    norm = norm[:, :, np.newaxis]

    # preview base64 PNG (uint8)
    _, buf = cv2.imencode(".png", processed.astype(np.uint8))
    b64 = base64.b64encode(buf).decode("utf-8")
    preview_dataurl = "data:image/png;base64," + b64

    return norm, preview_dataurl


# -------------------
# Inference (with normalization of numeric features)
# -------------------
def run_inference_and_debug(numeric_vector, imgB_arr, imgF_arr):
    """
    numeric_vector: numpy array shape (8,)
    imgB_arr, imgF_arr: arrays shape (H,W,1) normalized 0..1
    The function sets tensors in the order expected by the interpreter based on input_details.
    It also applies the exact normalization formulas you specified before.
    Returns: result_dict, normalized_vector
    """
    numeric_vector = numeric_vector.astype(np.float32).copy()
    print("Numeric before normalization:", numeric_vector.tolist())

    try:
        numeric_vector[0] = (numeric_vector[0] - 35) / (95 - 35)
        numeric_vector[1] = (numeric_vector[1] - 200) / (1500 - 200)
        numeric_vector[2] = (numeric_vector[2]) / 15.0
        numeric_vector[3] = (numeric_vector[3] - 0.61) / (1.057 - 0.61)
        numeric_vector[4] = (numeric_vector[4] - 0.608) / (1.01 - 0.608)
        numeric_vector[5] = numeric_vector[5]
        numeric_vector[6] = (numeric_vector[6]) / 133.53
        numeric_vector[7] = (numeric_vector[7]) / 5009.43
    except Exception as e:
        raise RuntimeError(f"Normalization failed: {e}")

    print("Numeric after normalization:", numeric_vector.tolist())

    img_count = 0
    for i, d in enumerate(input_details):
        idx = d["index"]
        shape = d.get("shape", [])
        if hasattr(shape, "__len__") and len(shape) == 2:
            vec = numeric_vector.astype(d["dtype"])[None, :]
            print(f"Setting numeric tensor at idx {idx}, shape set: {vec.shape}, dtype {d['dtype']}")
            interpreter.set_tensor(idx, vec)
        elif hasattr(shape, "__len__") and len(shape) == 4:
            if img_count == 0:
                arr = imgB_arr
            else:
                arr = imgF_arr
            arr_to_set = arr.astype(np.float32)[None, :, :, :]
            print(f"Setting image tensor #{img_count} at idx {idx}, shape set: {arr_to_set.shape}, dtype {d['dtype']}")
            interpreter.set_tensor(idx, arr_to_set)
            img_count += 1
        else:
            raise RuntimeError(f"Unrecognized input shape: {shape}")

    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])
    out = np.squeeze(out)
    cls = int(np.argmax(out))
    prob = float(out[cls]) if out.size > 0 else float("nan")
    label = CLASS_LABELS[cls] if cls < len(CLASS_LABELS) else f"Class {cls}"

    result = {"label": label, "prob": prob, "raw": out.tolist()}
    return result, numeric_vector


# -------------------
# Routes
# -------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    print("🔸 Raw JSON received:", bool(data))
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    nums = data.get("numbers", [])
    if not isinstance(nums, list) or len(nums) != 8:
        return jsonify({"error": "Need 8 numeric features (numbers)"}), 400
    try:
        numeric_vector = np.array([float(x) if x != "" else 0.0 for x in nums], dtype=np.float32)
    except Exception as e:
        return jsonify({"error": f"Numeric parsing failed: {e}"}), 400

    imageB_b64 = data.get("imageB")
    imageF_b64 = data.get("imageF")
    if not imageB_b64 or not imageF_b64:
        return jsonify({"error": "Both imageB and imageF must be provided"}), 400

    try:
        imgB_bytes = base64.b64decode(imageB_b64.split(",")[1])
        imgF_bytes = base64.b64decode(imageF_b64.split(",")[1])
    except Exception as e:
        return jsonify({"error": f"Failed to decode base64 images: {e}"}), 400

    H, W = required_image_size()
    try:
        arrB, previewB = preprocess_image_bytes(imgB_bytes, target_size=(H, W))
        arrF, previewF = preprocess_image_bytes(imgF_bytes, target_size=(H, W))
    except Exception as e:
        return jsonify({"error": f"Image preprocessing failed: {e}"}), 400

    if np.mean(arrB) < 0.01 or np.mean(arrF) < 0.01:
        return jsonify({"error": "Images appear too dark or invalid (mean pixel very small)"}), 400

    try:
        result, normalized_vector = run_inference_and_debug(numeric_vector, arrB, arrF)
    except Exception as e:
        print("❌ Inference exception:", e)
        return jsonify({"error": f"Inference failed: {e}"}), 500

    debug = {
        "numbers_raw": [float(x) for x in nums],
        "numbers_normalized": normalized_vector.tolist(),
        "imgB_shape": list(arrB.shape),
        "imgF_shape": list(arrF.shape),
    }

    response = {
        "result": result,
        "previewB": previewB,
        "previewF": previewF,
        "normalized": normalized_vector.tolist(),
        "debug": debug
    }

    response["label"] = result["label"]
    response["prob"] = result["prob"]

    print("🔸 Prediction result:", result)
    return jsonify(response)

@app.route("/healthz")
def healthz():
    return "ok"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
