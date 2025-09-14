# app.py
import io
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
model = load_model("model.h5")

def pad_to_square(image, fill=128):
    """پد کردن تصویر به مربع با رنگ خاکستری"""
    h, w = image.shape[:2]
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(image, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(fill, fill, fill))

def preprocess_image(data_url):
    """دریافت dataURL -> برگرداندن تصویر آماده مدل و preview"""
    # decode
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # grayscale + blur + threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # square pad (before resize)
    squared = pad_to_square(thresh)

    # resize to 256×256
    resized = cv2.resize(squared, (256, 256))

    # prepare for model (normalize)
    arr = resized.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)  # (256, 256, 1)
    arr = np.expand_dims(arr, axis=0)   # (1, 256, 256, 1)

    # preview (base64)
    _, buffer = cv2.imencode(".png", resized)
    preview_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")

    return arr, preview_b64

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        imgB_url = data.get("imageB")
        imgF_url = data.get("imageF")
        nums = data.get("numbers", [])

        # preprocess images
        arrB, previewB = preprocess_image(imgB_url)
        arrF, previewF = preprocess_image(imgF_url)

        # numeric inputs
        numeric = []
        for n in nums:
            try:
                numeric.append(float(n))
            except:
                numeric.append(0.0)
        numeric = np.array(numeric, dtype="float32").reshape(1, -1)

        # concatenate images + numbers → بسته به مدل شما
        # فرض: مدل ورودی‌های جدا می‌گیرد: [imgB, imgF, numeric]
        preds = model.predict([arrB, arrF, numeric])
        prob = float(preds[0][0])
        label = "Good" if prob >= 0.5 else "Bad"

        return jsonify({
            "result": {"label": label, "prob": prob},
            "previewB": previewB,
            "previewF": previewF,
            "normalized": numeric.tolist(),
            "debug": {"shapeB": arrB.shape, "shapeF": arrF.shape}
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
