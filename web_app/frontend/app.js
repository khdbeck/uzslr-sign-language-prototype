const BUFFER_SIZE = 32;
const CONFIDENCE_THRESHOLD = 0.45;
const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/infer`;

const N_FACE = 468;
const N_POSE = 33;
const N_HAND = 21;
const HAND_CONNECTIONS_LIST = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]
];

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const context = overlay.getContext('2d');
const cameraWrapper = document.getElementById('cameraWrapper');
const cameraEmpty = document.getElementById('cameraEmpty');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const bufferBar = document.getElementById('bufferBar');
const predictionIdle = document.getElementById('predictionIdle');
const predictionResult = document.getElementById('predictionResult');
const predLabel = document.getElementById('predLabel');
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
let gestureLocked = false;
let latestSign = null;
let currentVideoUrl = null;
let reconnectTimer = null;
const history = [];
const replySigns = [];

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
    setStatus('Камера готова — покажите обе руки', 'ready');
  });

  socket.addEventListener('message', (event) => {
    waitingForResponse = false;
    const data = JSON.parse(event.data);
    const progress = data.bufferFull ? 100 : Math.min(100, (data.bufferSize / BUFFER_SIZE) * 100);
    bufferBar.style.width = `${progress}%`;

    if (!data.handsVisible) {
      gestureLocked = false;
      cameraWrapper.classList.remove('tracking');
      setStatus('Покажите обе руки в кадре', 'ready');
      return;
    }

    cameraWrapper.classList.add('tracking');
    setStatus(data.bufferFull ? 'Анализируем жест…' : 'Записываем движение…', 'active');

    if (!data.prediction || gestureLocked) return;
    gestureLocked = true;
    const confidence = Number(data.confidence || 0);

    if (confidence < CONFIDENCE_THRESHOLD) {
      showUncertainResult(confidence);
      return;
    }

    showPrediction(data.prediction, confidence);
  });

  socket.addEventListener('close', () => {
    socketReady = false;
    waitingForResponse = false;
    setStatus('Восстанавливаем соединение…', 'connecting');
    reconnectTimer = window.setTimeout(connectSocket, 2000);
  });

  socket.addEventListener('error', () => {
    setStatus('Нет соединения с распознаванием', 'error');
  });
}

function showPrediction(sign, confidence) {
  latestSign = sign;
  const label = window.isoraSignLabel(sign);
  const percent = Math.round(confidence * 100);

  predictionIdle.classList.add('hidden');
  predictionResult.classList.remove('hidden');
  predictionResult.classList.remove('uncertain');
  predLabel.textContent = label;
  confPct.textContent = `${percent}%`;
  confBar.style.width = `${percent}%`;
  viewSignLink.href = `signs.html#${encodeURIComponent(sign)}`;
  confirmBtn.disabled = false;
  confirmStatus.textContent = '';

  if (history[0]?.sign !== sign) {
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
  predLabel.textContent = 'Не удалось распознать';
  confPct.textContent = `${percent}%`;
  confBar.style.width = `${percent}%`;
  viewSignLink.href = 'signs.html';
  confirmBtn.disabled = true;
}

function renderHistory() {
  if (!history.length) {
    historyList.innerHTML = '<span class="history-empty">История пока пуста</span>';
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
});

confirmBtn.addEventListener('click', () => {
  if (!latestSign) return;
  confirmStatus.textContent = `Перевод «${window.isoraSignLabel(latestSign)}» подтверждён`;
});

function buildLandmarkVector(results) {
  const vector = new Float32Array((N_FACE * 3) + (N_POSE * 4) + (N_HAND * 3 * 2));
  let offset = 0;

  const append = (landmarks, count, dimensions) => {
    for (let index = 0; index < count; index += 1) {
      const point = landmarks?.[index];
      if (point) {
        vector[offset] = point.x || 0;
        vector[offset + 1] = point.y || 0;
        vector[offset + 2] = point.z || 0;
        if (dimensions === 4) vector[offset + 3] = point.visibility ?? 0;
      }
      offset += dimensions;
    }
  };

  append(results.faceLandmarks, N_FACE, 3);
  append(results.poseLandmarks, N_POSE, 4);
  append(results.rightHandLandmarks, N_HAND, 3);
  append(results.leftHandLandmarks, N_HAND, 3);
  return Array.from(vector);
}

function drawHand(landmarks, color) {
  if (!landmarks) return;
  context.strokeStyle = color;
  context.fillStyle = '#ffc43d';
  context.lineWidth = 3;
  context.lineCap = 'round';

  HAND_CONNECTIONS_LIST.forEach(([start, end]) => {
    context.beginPath();
    context.moveTo(landmarks[start].x * overlay.width, landmarks[start].y * overlay.height);
    context.lineTo(landmarks[end].x * overlay.width, landmarks[end].y * overlay.height);
    context.stroke();
  });

  landmarks.forEach((point) => {
    context.beginPath();
    context.arc(point.x * overlay.width, point.y * overlay.height, 3.5, 0, Math.PI * 2);
    context.fill();
  });
}

function onHolisticResults(results) {
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }

  context.clearRect(0, 0, overlay.width, overlay.height);
  drawHand(results.leftHandLandmarks, '#ffffff');
  drawHand(results.rightHandLandmarks, '#ffffff');

  if (!socketReady || waitingForResponse) return;
  const hasBothHands = Boolean(results.leftHandLandmarks && results.rightHandLandmarks);
  waitingForResponse = true;
  socket.send(JSON.stringify({
    landmarks: buildLandmarkVector(results),
    hasBothHands
  }));
}

async function startCamera() {
  if (!window.Holistic || !window.Camera) {
    cameraEmpty.querySelector('strong').textContent = 'Модуль камеры не загрузился';
    cameraEmpty.querySelector('span').textContent = 'Проверьте интернет-соединение и обновите страницу';
    setStatus('Не удалось загрузить MediaPipe', 'error');
    return;
  }

  try {
    const holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });
    holistic.setOptions({
      modelComplexity: 0,
      smoothLandmarks: true,
      enableSegmentation: false,
      refineFaceLandmarks: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
    holistic.onResults(onHolisticResults);

    const camera = new Camera(video, {
      onFrame: async () => holistic.send({ image: video }),
      width: 640,
      height: 480
    });
    await camera.start();
    cameraEmpty.classList.add('hidden');
  } catch (error) {
    cameraEmpty.classList.remove('hidden');
    cameraEmpty.querySelector('strong').textContent = 'Камера недоступна';
    cameraEmpty.querySelector('span').textContent = 'Разрешите доступ в настройках браузера и обновите страницу';
    setStatus('Нет доступа к камере', 'error');
  }
}

function renderReplySequence() {
  if (!replySigns.length) {
    replySequence.innerHTML = '<span>Выберите фразу выше</span>';
  } else {
    replySequence.innerHTML = replySigns.map((sign, index) => `
      <button type="button" data-remove-index="${index}" aria-label="Удалить ${window.isoraSignLabel(sign)}">
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
    replyStatus.textContent = 'Для демонстрации можно выбрать до 8 жестов';
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
  replyStatus.textContent = 'Собираем видеожесты…';

  try {
    const response = await fetch('/api/produce-sign-video', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ signs: replySigns })
    });
    if (!response.ok) {
      const details = await response.json().catch(() => ({}));
      throw new Error(details.detail || 'Не удалось собрать ответ');
    }

    if (currentVideoUrl) URL.revokeObjectURL(currentVideoUrl);
    currentVideoUrl = URL.createObjectURL(await response.blob());
    replyVideo.src = currentVideoUrl;
    avatarPlaceholder.classList.add('hidden');
    replyVideo.classList.remove('hidden');
    replyStatus.textContent = 'Ответ готов';
    await replyVideo.play().catch(() => {});
  } catch (error) {
    replyStatus.textContent = error.message;
  } finally {
    generateReplyBtn.classList.remove('loading');
    generateReplyBtn.disabled = !replySigns.length;
  }
});

connectSocket();
startCamera();
