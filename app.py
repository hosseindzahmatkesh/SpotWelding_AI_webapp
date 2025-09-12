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
    # ما همیشه می‌خوایم 256x256 باشه
    return 256, 256


def preprocess_image(img_bytes, target_size, return_stage="final"):
    """
    Decode base64 image → grayscale → blur + threshold → resize → normalize
    return_stage:
        "final" → آماده برای مدل
        "preview" → قبل از normalize (uint8 برای نمایش)
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # blur + threshold
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)

    resized = cv2.resize(thresh, target_size)

    if return_stage == "preview":
        return resized  # uint8
    else:
        norm = resized / 255.0
        norm = norm[:, :, np.newaxis]
        return norm



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
            # -------------------
            # Normalization
            # -------------------
            numeric_vector[0] = (numeric_vector[0] - 35) / (95 - 35)       # Pressure
            numeric_vector[1] = (numeric_vector[1] - 200) / (1500 - 200)   # Welding Time
            numeric_vector[2] = numeric_vector[2] / 15                     # Angle
            numeric_vector[3] = (numeric_vector[3] - 0.61) / (1.057 - 0.61) # Thickness A
            numeric_vector[4] = (numeric_vector[4] - 0.608) / (1.01 - 0.608) # Thickness B
            numeric_vector[5] = numeric_vector[5]                           # Material (assume already categorical/encoded)
            numeric_vector[6] = numeric_vector[6] / 133.53                  # Force
            numeric_vector[7] = numeric_vector[7] / 5009.43                 # Current

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


@app.route("/preview", methods=["POST"])
def preview():
    data = request.json
    try:
        raw = base64.b64decode(data["image"].split(",")[1])
    except Exception:
        return jsonify({"error": "invalid image payload"}), 400

    H, W = required_image_size()
    arr = preprocess_image(raw, (W, H), return_stage="preview")

    # arr → base64
    _, buf = cv2.imencode(".png", arr)
    b64 = base64.b64encode(buf).decode("utf-8")
    return jsonify({"processed": "data:image/png;base64," + b64})



@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        imgB_data = base64.b64decode(data["imageB"].split(",")[1])
        imgF_data = base64.b64decode(data["imageF"].split(",")[1])
    except Exception:
        return jsonify({"error": "invalid image payload"}), 400

    H, W = required_image_size()
    arrB = preprocess_image(imgB_data, (W, H))
    arrF = preprocess_image(imgF_data, (W, H))

    nums = data.get("numbers", [])
    if not isinstance(nums, list) or len(nums) != 8:
        return jsonify({"error": "need 8 numeric features"}), 400
    try:
        numeric_vector = np.array([float(x) if x != "" else 0.0 for x in nums], dtype=np.float32)
    except Exception:
        return jsonify({"error": "non-numeric feature detected"}), 400

    result = run_inference(numeric_vector, arrB, arrF)
    return jsonify(result)


@app.route("/healthz")
def healthz():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
