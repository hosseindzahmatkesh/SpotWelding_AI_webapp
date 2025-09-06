const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const overlay = document.getElementById('overlay');
const statusEl = document.getElementById('status');

let stream = null;
let modelW = window.MODEL_W || 256;
let modelH = window.MODEL_H || 256;

function getNumbers() {
  const ids = ["n1","n2","n3","n4","n5","n6","n7","n8"];
  return ids.map(id => document.getElementById(id).value.trim());
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
    video.srcObject = stream;
    statusEl.textContent = "دوربین فعال است.";
  } catch (e) {
    console.error(e);
    statusEl.textContent = "خطا در دسترسی به دوربین: " + e.message;
    alert("اجازه دسترسی به دوربین را فعال کنید. روی موبایل نیاز به HTTPS دارید.");
  }
}
function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  statusEl.textContent = "دوربین متوقف شد.";
}
async function captureAndPredict() {
  if (!stream) { alert("دوربین غیرفعال است."); return; }
  // Draw current frame to a canvas sized to model input
  canvas.width = modelW;
  canvas.height = modelH;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, modelW, modelH);
  const dataURL = canvas.toDataURL("image/png");
  await sendForPredict(dataURL);
}
async function predictCurrentFrame() {
  if (!stream) { alert("دوربین غیرفعال است."); return; }
  return captureAndPredict();
}
async function sendForPredict(dataURL) {
  try {
    const numbers = getNumbers();
    overlay.textContent = "در حال پردازش…";
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataURL, numbers })
    });
    const js = await res.json();
    if (js.error) throw new Error(js.error);
    overlay.textContent = `نتیجه: ${js.label} | دقت: ${(js.prob*100).toFixed(1)}%`;
  } catch (e) {
    console.error(e);
    overlay.textContent = "خطا در پیش‌بینی";
    alert("Prediction error: " + e.message);
  }
}

document.getElementById('btn-capture').addEventListener('click', captureAndPredict);
document.getElementById('btn-predict').addEventListener('click', predictCurrentFrame);
document.getElementById('btn-stop').addEventListener('click', stopCamera);

// Auto-start
startCamera();
