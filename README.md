# SpotWelding Web AI Demo

## ساخت و اجرا (لوکال)
1) مدل خود را با نام `model.tflite` کنار `app.py` قرار دهید.
2) محیط پایتون بسازید و وابستگی‌ها را نصب کنید:
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```
3) سرور را اجرا کنید:
```bash
python app.py
```
4) در مرورگر به `http://127.0.0.1:5000` بروید. برای دسترسی از موبایل داخل شبکه، با IP سیستم خودتان باز کنید (مثلاً `http://192.168.1.10:5000`). توجه: برای فعال شدن دوربین روی موبایل، HTTPS لازم است (یا از سرویس‌های میزبانی HTTPS استفاده کنید).

## دیپلوی روی Render
- یک سرویس Web جدید بسازید و مخزن را وصل کنید.
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app`
- فایل‌ها: `app.py`, `requirements.txt`, پوشه‌های `templates`, `static`, و `model.tflite` را آپلود کنید.

## API
- `POST /predict` با بدنه:
```json
{
  "image": "data:image/png;base64,...",
  "numbers": [P, WT, Ang, ThA, ThB, MaterialCode, Force, Current]
}
```
خروجی:
```json
{"label":"Good","prob":0.91,"raw":[...]}
```
<<<<<<< HEAD
"# spotwelding_webapp" 
"# spotwelding_webapp" 
=======
"# spotwelding_webapp" 
"# spotwelding_webapp" 
>>>>>>> d7e71d8c (first commit)
