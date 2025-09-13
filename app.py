import gdown
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import base64, io

# -------------------
# Download model if missing
# -------------------
MODEL_PATH = "model.tflite"
MODEL_URL = "https://drive.google.com/uc?id=1-e7UBpGkYgQlrfrOfpP9TtJ7UZt8A2rx"
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -------------------
# Load TFLite
# -------------------
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    import tensorflow as tf
    tflite = tf.lite

app = Flask(__name__)

CLASS_LABELS = ["Bad", "Good", "Explode"]
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# -------------------
# Utils
# -------------------
def required_image_size():
    for d in input_details:
        shape = d.get("shape", [])
        if hasattr(shape, "__len__") and len(shape) == 4:
            return int(shape[1]), int(shape[2])
    return 256, 256


def preprocess_image(img_bytes, target_size):
    """
    Decode base64 image, apply circle mask (inside: real, outside: gray),
    then blur + threshold, resize, normalize.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    # --- Circle mask: match overlay ---
    mask = np.zeros_like(gray, dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = 125
    cv2.circle(mask, center, radius, 255, -1)

    gray_bg = np.full_like(gray, 128)
    masked = np.where(mask == 255, gray, gray_bg)

    # --- Preprocess ---
    blur = cv2.GaussianBlur(masked, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)

    # Resize
    resized = cv2.resize(thresh, target_size)

    # Normalize
    norm = resized / 255.0
    norm = norm[:, :, np.newaxis]  # add channel

    # Also return a preview image (base64)
    arr_uint8 = resized.astype(np.uint8)
    _, buf = cv2.imencode(".png", arr_uint8)
    b64 = base64.b64encode(buf).decode("utf-8")

    return norm, "data:image/png;base64," + b64


def run_inference(numeric_vector, imgB, imgF):
    """
    Order must be:
    1) numeric_vector
    2) imageB
    3) imageF
    """
    for i, d in enumerate(input_details):
        idx = d["index"]
        shape = d.get("shape", [])

        if len(shape) == 2:  # numeric input
            # -------- Normalization --------
            print("Numeric input before normalization:", numeric_vector.tolist())
            numeric_vector[0] = (numeric_vector[0] - 35) / (95 - 35)
            numeric_vector[1] = (numeric_vector[1] - 200) / (1500 - 200)
            numeric_vector[2] = (numeric_vector[2]) / 15
            numeric_vector[3] = (numeric_vector[3] - 0.61) / (1.057 - 0.61)
            numeric_vector[4] = (numeric_vector[4] - 0.608) / (1.01 - 0.608)
            numeric_vector[5] = numeric_vector[5]  # no normalization
            numeric_vector[6] = (numeric_vector[6]) / 133.53
            numeric_vector[7] = (numeric_vector[7]) / 5009.43
            print("Numeric input after normalization:", numeric_vector.tolist())

            vec = numeric_vector.astype(d["dtype"])[None, :]
            interpreter.set_tensor(idx, vec)

        elif len(shape) == 4:  # image input
            if i == 1:   # بعد از عددی → این imgB است
                arr = imgB
            else:        # بعدی → imgF است
                arr = imgF
            arr = arr.astype(np.float32)[None, :, :, :]
            interpreter.set_tensor(idx, arr)

    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])
    out = np.squeeze(out)
    cls = int(np.argmax(out))
    prob = float(out[cls])
    label = CLASS_LABELS[cls] if cls < len(CLASS_LABELS) else f"Class {cls}"
    return {"label": label, "prob": prob, "raw": out.tolist()}


# -------------------
# Routes
# -------------------
@app.route("/")
def index():
    h, w = required_image_size()
    return render_template("index.html", imgw=w, imgh=h, labels=CLASS_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        imgB_data = base64.b64decode(data["imageB"].split(",")[1])
        imgF_data = base64.b64decode(data["imageF"].split(",")[1])
    except Exception:
        return jsonify({"error": "invalid image payload"}), 400

    H, W = required_image_size()
    arrB, previewB = preprocess_image(imgB_data, (W, H))
    arrF, previewF = preprocess_image(imgF_data, (W, H))

    # Quick quality check
    if np.mean(arrB) < 0.05 or np.mean(arrF) < 0.05:
        return jsonify({"error": "Image too dark, invalid capture"}), 400

    nums = data.get("numbers", [])
    if not isinstance(nums, list) or len(nums) != 8:
        return jsonify({"error": "need 8 numeric features"}), 400
    try:
        numeric_vector = np.array([float(x) if x != "" else 0.0 for x in nums], dtype=np.float32)
    except Exception:
        return jsonify({"error": "non-numeric feature detected"}), 400

    result = run_inference(numeric_vector, arrB, arrF)

    return jsonify({
        "result": result,
        "previewB": previewB,
        "previewF": previewF
    })


@app.route("/healthz")
def healthz():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
