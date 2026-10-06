let BUFFER_SIZE = 32;
let CONFIDENCE_THRESHOLD = 0.1;
let SAMPLE_FPS = 15;
const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/infer`;

const video = document.getElementById('video');
const cameraWrapper = document.getElementById('cameraWrapper');
const cameraEmpty = document.getElementById('cameraEmpty');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const bufferBar = document.getElementById('bufferBar');
const predictionIdle = document.getElementById('predictionIdle');
const predictionResult = document.getElementById('predictionResult');
const predLabel = document.getElementById('predLabel');
const resultKicker = document.getElementById('resultKicker');
const confBar = document.getElementById('confBar');
const confPct = document.getElementById('confPct');
const viewSignLink = document.getElementById('viewSignLink');
const historyList = document.getElementById('historyList');
const clearBtn = document.getElementById('clearBtn');
const confirmBtn = document.getElementById('confirmBtn');
const confirmStatus = document.getElementById('confirmStatus');
const quickReplies = document.getElementById('quickReplies');
const replySequence = document.getElementById('replySequence');
const generateReplyBtn = document.getElementById('generateReplyBtn');
const clearReplyBtn = document.getElementById('clearReplyBtn');
const avatarPlaceholder = document.getElementById('avatarPlaceholder');
const replyVideo = document.getElementById('replyVideo');
const replyStatus = document.getElementById('replyStatus');

let socket;
let socketReady = false;
let waitingForResponse = false;
let latestSign = null;
let currentVideoUrl = null;
let reconnectTimer = null;
const history = [];
const replySigns = [];
let cameraStream = null;
let clipTesting = false;
let captureTimer = null;
const captureCanvas = document.createElement('canvas');
const captureContext = captureCanvas.getContext('2d');
const videoFile = document.getElementById('videoFile');
const videoTestStatus = document.getElementById('videoTestStatus');
const resumeCameraBtn = document.getElementById('resumeCameraBtn');
const top5List = document.getElementById('top5List');

function setStatus(message, state = 'connecting') {
  statusText.textContent = message;
  statusDot.className = `status-dot ${state}`;
}

function connectSocket() {
  window.clearTimeout(reconnectTimer);
  socket = new WebSocket(WS_URL);

  socket.addEventListener('open', () => {
    socketReady = true;
    waitingForResponse = false;
    setStatus('Slovo tayyor — ishorani kameraga ko‘rsating', 'ready');
  });

  socket.addEventListener('message', (event) => {
    waitingForResponse = false;
    const data = JSON.parse(event.data);
    if (clipTesting) return;
    if (data.error) {
      setStatus(data.error, 'error');
      return;
    }
    const progress = data.bufferFull ? 100 : Math.min(100, (data.bufferSize / BUFFER_SIZE) * 100);
    bufferBar.style.width = `${progress}%`;

    cameraWrapper.classList.add('tracking');
    setStatus(data.bufferFull ? 'Ishora tahlil qilinmoqda…' : 'Harakat yozib olinmoqda…', 'active');
    if (data.bufferFull) displayResult(data);
  });

  socket.addEventListener('close', () => {
    socketReady = false;
    waitingForResponse = false;
    setStatus('Ulanish tiklanmoqda…', 'connecting');
    reconnectTimer = window.setTimeout(connectSocket, 2000);
  });

  socket.addEventListener('error', () => {
    setStatus("Tanib olish tizimi bilan aloqa yo'q", 'error');
  });
}

function showPrediction(sign, confidence, noEvent = false) {
  latestSign = noEvent ? null : sign;
  const label = noEvent ? 'Ishora yo‘q' : window.isoraSignLabel(sign);
  const percent = Math.round(confidence * 100);

  predictionIdle.classList.add('hidden');
  predictionResult.classList.remove('hidden');
  predictionResult.classList.remove('uncertain');
  predLabel.textContent = label;
  resultKicker.textContent = 'Eng ehtimolli natija';
  confPct.textContent = `${percent}%`;
  confBar.style.width = `${percent}%`;
  viewSignLink.href = noEvent ? 'slovo.html' : `slovo.html#${encodeURIComponent(sign)}`;
  viewSignLink.textContent = 'Ishoralar lug‘atida ko‘rish';
  confirmBtn.disabled = noEvent;
  confirmStatus.textContent = '';

  if (!noEvent && history[0]?.sign !== sign) {
    history.unshift({ sign, label, confidence: percent });
    if (history.length > 6) history.pop();
    renderHistory();
  }
}

function showUncertainResult(confidence) {
  latestSign = null;
  const percent = Math.round(confidence * 100);
  predictionIdle.classList.add('hidden');
  predictionResult.classList.remove('hidden');
  predictionResult.classList.add('uncertain');
  predLabel.textContent = "Ishorani aniqlab bo'lmadi";
  resultKicker.textContent = 'Noaniq natija';
  confPct.textContent = `${percent}%`;
  confBar.style.width = `${percent}%`;
  viewSignLink.href = 'slovo.html';
  confirmBtn.disabled = true;
  confirmStatus.textContent = '';
}

function renderHistory() {
  if (!history.length) {
    historyList.innerHTML = '<span class="history-empty">Tarix hozircha bo\'sh</span>';
    return;
  }

  historyList.innerHTML = history.map((item, index) => `
    <div class="history-item">
      <span>${history.length - index}</span>
      <strong>${item.label}</strong>
      <small>${item.confidence}%</small>
    </div>
  `).join('');
}

clearBtn.addEventListener('click', () => {
  history.length = 0;
  latestSign = null;
  renderHistory();
  predictionResult.classList.add('hidden');
  predictionResult.classList.remove('uncertain');
  predictionIdle.classList.remove('hidden');
  confirmBtn.disabled = true;
  confirmStatus.textContent = '';
  top5List.replaceChildren();
  if (socketReady) socket.send('reset');
});

confirmBtn.addEventListener('click', () => {
  if (!latestSign) return;
  confirmStatus.textContent = `«${window.isoraSignLabel(latestSign)}» tarjimasi tasdiqlandi`;
});

function displayResult(data) {
  const confidence = Number(data.confidence || 0);
  top5List.replaceChildren();
  for (const candidate of data.top5 || []) {
    const item = document.createElement('li');
    const label = candidate.label === '---' ? 'Ishora yo‘q' : candidate.label;
    item.textContent = `${label} — ${(candidate.confidence * 100).toFixed(1)}%`;
    top5List.append(item);
  }
  if (!data.noEvent && confidence < CONFIDENCE_THRESHOLD) {
    showUncertainResult(confidence);
  } else {
    showPrediction(data.prediction, confidence, data.noEvent);
  }
}

function captureFrame() {
  if (!cameraStream || clipTesting || !socketReady || waitingForResponse || video.readyState < 2) return;
  const ratio = Math.min(1, 640 / video.videoWidth);
  captureCanvas.width = Math.round(video.videoWidth * ratio);
  captureCanvas.height = Math.round(video.videoHeight * ratio);
  // CSS mirrors the preview only. Model receives the original camera orientation.
  captureContext.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  waitingForResponse = true;
  captureCanvas.toBlob((blob) => {
    if (!blob || clipTesting || !socketReady || socket.readyState !== WebSocket.OPEN) {
      waitingForResponse = false;
      return;
    }
    socket.send(blob);
  }, 'image/jpeg', 0.85);
}

async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false
    });
    video.srcObject = cameraStream;
    await video.play();
    cameraEmpty.classList.add('hidden');
    window.clearInterval(captureTimer);
    captureTimer = window.setInterval(captureFrame, 1000 / SAMPLE_FPS);
  } catch (error) {
    cameraEmpty.classList.remove('hidden');
    cameraEmpty.querySelector('strong').textContent = 'Kamera mavjud emas';
    cameraEmpty.querySelector('span').textContent = 'Kameraga ruxsat bering yoki pastdan video yuklang';
    setStatus("Kameraga ruxsat yo'q", 'error');
  }
}

videoFile.addEventListener('change', async () => {
  const file = videoFile.files[0];
  if (!file) return;
  if (file.size > 50 * 1024 * 1024) {
    videoTestStatus.textContent = 'Video 50 MB dan kichik bo‘lishi kerak';
    return;
  }
  clipTesting = true;
  resumeCameraBtn.classList.remove('hidden');
  videoFile.disabled = true;
  videoTestStatus.textContent = 'Video tahlil qilinmoqda…';
  setStatus('Video tekshiruvi — kamera tahlili to‘xtatilgan', 'ready');
  const form = new FormData();
  form.append('file', file);
  try {
    const response = await fetch('/api/infer-video', { method: 'POST', body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Video tekshiruvi bajarilmadi');
    displayResult(result);
    videoTestStatus.textContent = `${file.name}: ${result.noEvent ? 'ishora yo‘q' : result.prediction} (${(result.confidence * 100).toFixed(1)}%)`;
  } catch (error) {
    videoTestStatus.textContent = error.message;
  } finally {
    videoFile.disabled = false;
    videoFile.value = '';
  }
});

resumeCameraBtn.addEventListener('click', () => {
  clipTesting = false;
  resumeCameraBtn.classList.add('hidden');
  videoTestStatus.textContent = '';
  if (socketReady) socket.send('reset');
  if (!cameraStream) startCamera();
});

window.addEventListener('pagehide', () => {
  window.clearInterval(captureTimer);
  window.clearTimeout(reconnectTimer);
  cameraStream?.getTracks().forEach((track) => track.stop());
  socket?.close();
});

function renderReplySequence() {
  if (!replySigns.length) {
    replySequence.innerHTML = '<span>Yuqoridagi iboralardan birini tanlang</span>';
  } else {
    replySequence.innerHTML = replySigns.map((sign, index) => `
      <button type="button" data-remove-index="${index}" aria-label="Olib tashlash: ${window.isoraSignLabel(sign)}">
        ${window.isoraSignLabel(sign)}<b aria-hidden="true">×</b>
      </button>
    `).join('');
  }

  generateReplyBtn.disabled = !replySigns.length;
  clearReplyBtn.disabled = !replySigns.length;
}

quickReplies.addEventListener('click', (event) => {
  const button = event.target.closest('[data-sign]');
  if (!button) return;
  const sign = button.dataset.sign;
  if (replySigns.length >= 8) {
    replyStatus.textContent = "Namoyish uchun ko'pi bilan 8 ta ishorani tanlash mumkin";
    return;
  }
  replySigns.push(sign);
  replyStatus.textContent = '';
  renderReplySequence();
});

replySequence.addEventListener('click', (event) => {
  const button = event.target.closest('[data-remove-index]');
  if (!button) return;
  replySigns.splice(Number(button.dataset.removeIndex), 1);
  renderReplySequence();
});

clearReplyBtn.addEventListener('click', () => {
  replySigns.length = 0;
  replyStatus.textContent = '';
  renderReplySequence();
});

generateReplyBtn.addEventListener('click', async () => {
  if (!replySigns.length) return;
  generateReplyBtn.disabled = true;
  generateReplyBtn.classList.add('loading');
  replyStatus.textContent = "Videoishoralar yig'ilmoqda…";

  try {
    const response = await fetch('/api/produce-sign-video', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ signs: replySigns })
    });
    if (!response.ok) {
      const details = await response.json().catch(() => ({}));
      throw new Error(details.detail || "Javobni yig'ib bo'lmadi");
    }

    if (currentVideoUrl) URL.revokeObjectURL(currentVideoUrl);
    currentVideoUrl = URL.createObjectURL(await response.blob());
    replyVideo.src = currentVideoUrl;
    avatarPlaceholder.classList.add('hidden');
    replyVideo.classList.remove('hidden');
    replyStatus.textContent = 'Javob tayyor';
    await replyVideo.play().catch(() => {});
  } catch (error) {
    replyStatus.textContent = error.message;
  } finally {
    generateReplyBtn.classList.remove('loading');
    generateReplyBtn.disabled = !replySigns.length;
  }
});

async function initializeRecognition() {
  try {
    const response = await fetch('/api/config');
    if (!response.ok) throw new Error('Tanib olish tizimi yuklanmadi');
    const config = await response.json();
    BUFFER_SIZE = config.bufferSize;
    SAMPLE_FPS = config.sampleFps;
    CONFIDENCE_THRESHOLD = config.confidenceThreshold;
    connectSocket();
    await startCamera();
  } catch (error) {
    setStatus(error.message, 'error');
  }
}
initializeRecognition();
