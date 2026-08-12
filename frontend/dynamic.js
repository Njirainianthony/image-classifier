/* ============================================================
   Model Bench — dynamic.js
   ------------------------------------------------------------
   This file wires up the UI. The one thing you need to change
   to go from "demo" to "real" is the runInference() function
   below — see the big comment block right above it.
   ============================================================ */

(() => {
  'use strict';

  /* ----------------------------------------------------------
     DOM references
  ---------------------------------------------------------- */
  const dropzone        = document.getElementById('dropzone');
  const fileInput        = document.getElementById('fileInput');
  const browseBtn        = document.getElementById('browseBtn');
  const cameraBtn         = document.getElementById('cameraBtn');
  const previewImg        = document.getElementById('previewImg');
  const dropzoneEmpty     = document.getElementById('dropzoneEmpty');
  const fileMeta           = document.getElementById('fileMeta');

  const resultsEmpty      = document.getElementById('resultsEmpty');
  const resultsBody        = document.getElementById('resultsBody');
  const loadingState        = document.getElementById('loadingState');
  const predictionList       = document.getElementById('predictionList');

  const verifySection      = document.getElementById('verifySection');
  const markCorrectBtn      = document.getElementById('markCorrect');
  const markIncorrectBtn     = document.getElementById('markIncorrect');
  const trueLabelWrap        = document.getElementById('trueLabelWrap');
  const trueLabelInput        = document.getElementById('trueLabelInput');
  const submitTrueLabelBtn     = document.getElementById('submitTrueLabel');
  const verifySaved             = document.getElementById('verifySaved');

  const statTotal       = document.getElementById('statTotal');
  const statAccuracy     = document.getElementById('statAccuracy');
  const resetStatsBtn     = document.getElementById('resetStats');

  const logBody         = document.getElementById('logBody');
  const logEmptyRow      = document.getElementById('logEmptyRow');
  const exportLogBtn      = document.getElementById('exportLog');

  /* ----------------------------------------------------------
     State
  ---------------------------------------------------------- */
  const STORAGE_KEY = 'modelbench_log_v1';

  let currentImage = {
    dataUrl: null,
    name: null
  };
  let currentPrediction = null; // { label, confidence, all: [...] }
  let log = loadLog();

  /* ----------------------------------------------------------
     ============================================================
     RUN INFERENCE — plug your model in here
     ============================================================

     This function must return a Promise that resolves to an array
     of predictions sorted by confidence, descending:

       [{ label: "golden retriever", confidence: 0.94 },
        { label: "labrador",         confidence: 0.04 },
        { label: "terrier",          confidence: 0.01 }]

     `imageEl` is the <img> element already showing the user's
     photo (already loaded, natural width/height available).

     -------- OPTION A: TensorFlow.js (model trained/exported
     from your notebook, e.g. via tensorflowjs_converter) --------

       1. Add this to index.html <head>:
          <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>

       2. Host your converted model folder (model.json + shards)
          next to these files, e.g. /model/model.json

       3. Replace the body of runInference with:

          const model = await getModel(); // see loadModelOnce() below
          const tensor = tf.browser.fromPixels(imageEl)
            .resizeNearestNeighbor([224, 224])   // match your notebook's input size
            .toFloat()
            .div(255.0)
            .expandDims();
          const output = await model.predict(tensor).data();
          tensor.dispose();

          return LABELS
            .map((label, i) => ({ label, confidence: output[i] }))
            .sort((a, b) => b.confidence - a.confidence);

     -------- OPTION B: a backend API (e.g. Flask/FastAPI serving
     the model straight from your notebook's saved weights) --------

          const formData = new FormData();
          formData.append('image', currentImage.file);
          const res = await fetch('https://your-api.example.com/predict', {
            method: 'POST',
            body: formData
          });
          const data = await res.json();
          return data.predictions; // shape it to match the format above

     Until you wire one of these in, this function returns
     randomized mock predictions so the UI is fully testable.
  ---------------------------------------------------------- */

  const PREDICT_ENDPOINT = 'http://127.0.0.1:8000/predict';

  async function runInference(imageEl) {
    // Calls the local model server. The server is expected to return:
    //   { "prediction": "car", "confidence": 0.85 }
    //
    // The image is sent as multipart/form-data under the field name
    // "image". If your endpoint expects a different field name, change
    // the formData.append() key below to match.

    const blob = await (await fetch(imageEl.src)).blob();
    const formData = new FormData();
    formData.append('file', blob, currentImage.name || 'upload.png');

    const res = await fetch(PREDICT_ENDPOINT, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      throw new Error(`Prediction server responded with ${res.status}`);
    }

    const data = await res.json();

    // Normalize the server's single-label response into the
    // { label, confidence } list format the UI renders.
    return [
      { label: data.prediction, confidence: data.confidence }
    ];
  }

  /* Example helper for Option A above — loads the tf.js model once
     and caches it, so you're not re-loading it on every prediction. */
  let _modelCache = null;
  async function loadModelOnce(modelUrl) {
    if (_modelCache) return _modelCache;
    _modelCache = await tf.loadLayersModel(modelUrl); // or loadGraphModel
    return _modelCache;
  }

  /* ----------------------------------------------------------
     File handling
  ---------------------------------------------------------- */
  browseBtn.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });

  ['dragenter', 'dragover'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add('is-dragover');
    });
  });
  ['dragleave', 'drop'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove('is-dragover');
    });
  });
  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // Camera capture: reuses the same file input via the `capture` attribute
  // on supporting mobile browsers. Falls back to normal file picking.
  cameraBtn.addEventListener('click', () => {
    fileInput.setAttribute('capture', 'environment');
    fileInput.click();
    setTimeout(() => fileInput.removeAttribute('capture'), 0);
  });

  function handleFile(file) {
    if (!file.type.startsWith('image/')) {
      fileMeta.textContent = 'That file doesn\u2019t look like an image — try a JPG, PNG, or WebP.';
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      currentImage.dataUrl = e.target.result;
      currentImage.name = file.name;

      previewImg.src = currentImage.dataUrl;
      previewImg.hidden = false;
      dropzoneEmpty.hidden = true;

      const kb = (file.size / 1024).toFixed(0);
      fileMeta.textContent = `${file.name} · ${kb} KB`;

      previewImg.onload = () => predict();
    };
    reader.readAsDataURL(file);
  }

  /* ----------------------------------------------------------
     Prediction flow
  ---------------------------------------------------------- */
  async function predict() {
    resultsEmpty.hidden = true;
    resultsBody.hidden = false;
    loadingState.hidden = false;
    predictionList.innerHTML = '';
    verifySection.hidden = true;
    verifySaved.hidden = true;
    trueLabelWrap.hidden = true;
    trueLabelInput.value = '';

    try {
      const predictions = await runInference(previewImg);
      currentPrediction = predictions;
      renderPredictions(predictions);
      verifySection.hidden = false;
    } catch (err) {
      predictionList.innerHTML = `<li style="color: var(--accent-coral); font-family: var(--font-mono); font-size: 12px;">Inference failed: ${escapeHtml(err.message || String(err))}</li>`;
    } finally {
      loadingState.hidden = true;
    }
  }

  function renderPredictions(predictions) {
    predictionList.innerHTML = '';
    predictions.slice(0, 5).forEach((p, i) => {
      const li = document.createElement('li');
      if (i === 0) li.classList.add('is-top');
      const pct = (p.confidence * 100).toFixed(1);
      li.innerHTML = `
        <span class="pred-rank">${i + 1}</span>
        <span class="pred-main">
          <span class="pred-label">${escapeHtml(p.label)}</span>
          <span class="pred-bar"><span class="pred-bar__fill" style="width:0%"></span></span>
        </span>
        <span class="pred-confidence">${pct}%</span>
      `;
      predictionList.appendChild(li);
      requestAnimationFrame(() => {
        li.querySelector('.pred-bar__fill').style.width = `${pct}%`;
      });
    });
  }

  /* ----------------------------------------------------------
     Verification / accuracy tracking
  ---------------------------------------------------------- */
  markCorrectBtn.addEventListener('click', () => {
    logResult(true, currentPrediction[0].label);
  });

  markIncorrectBtn.addEventListener('click', () => {
    trueLabelWrap.hidden = false;
    trueLabelInput.focus();
  });

  submitTrueLabelBtn.addEventListener('click', submitTrueLabel);
  trueLabelInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') submitTrueLabel();
  });

  function submitTrueLabel() {
    const val = trueLabelInput.value.trim();
    logResult(false, val || '(unspecified)');
  }

  function logResult(wasCorrect, trueLabel) {
    const entry = {
      id: Date.now(),
      thumb: currentImage.dataUrl,
      predicted: currentPrediction[0].label,
      confidence: currentPrediction[0].confidence,
      correct: wasCorrect,
      trueLabel: wasCorrect ? currentPrediction[0].label : trueLabel
    };
    log.unshift(entry);
    saveLog();
    renderLog();
    renderStats();

    verifySaved.hidden = false;
    verifySection.querySelector('.verify__actions').style.display = 'none';
    trueLabelWrap.hidden = true;
  }

  /* ----------------------------------------------------------
     Log table
  ---------------------------------------------------------- */
  function renderLog() {
    logBody.innerHTML = '';
    if (log.length === 0) {
      logBody.appendChild(logEmptyRow);
      return;
    }
    log.forEach((entry, i) => {
      const tr = document.createElement('tr');
      const tagClass = entry.correct ? 'tag--correct' : 'tag--incorrect';
      const tagText = entry.correct ? 'correct' : 'incorrect';
      tr.innerHTML = `
        <td>${log.length - i}</td>
        <td><img class="log-thumb" src="${entry.thumb}" alt=""></td>
        <td>${escapeHtml(entry.predicted)}</td>
        <td>${(entry.confidence * 100).toFixed(1)}%</td>
        <td><span class="tag ${tagClass}">${tagText}</span></td>
        <td>${escapeHtml(entry.trueLabel)}</td>
      `;
      logBody.appendChild(tr);
    });
  }

  function renderStats() {
    const total = log.length;
    const correct = log.filter(e => e.correct).length;
    statTotal.textContent = total;
    statAccuracy.textContent = total === 0 ? '—' : `${((correct / total) * 100).toFixed(0)}%`;
  }

  resetStatsBtn.addEventListener('click', () => {
    if (log.length === 0) return;
    if (!confirm('Clear all logged results for this session? This can\u2019t be undone.')) return;
    log = [];
    saveLog();
    renderLog();
    renderStats();
  });

  exportLogBtn.addEventListener('click', () => {
    if (log.length === 0) return;
    const rows = [['#', 'predicted_label', 'confidence', 'result', 'true_label']];
    log.forEach((e, i) => {
      rows.push([
        log.length - i,
        e.predicted,
        e.confidence.toFixed(4),
        e.correct ? 'correct' : 'incorrect',
        e.trueLabel
      ]);
    });
    const csv = rows.map(r => r.map(csvEscape).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model-bench-log.csv';
    a.click();
    URL.revokeObjectURL(url);
  });

  function csvEscape(val) {
    const s = String(val);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }

  /* ----------------------------------------------------------
     Persistence (session log only — survives reloads via
     localStorage, resettable at any time)
  ---------------------------------------------------------- */
  function loadLog() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  function saveLog() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(log));
    } catch {
      // storage full or unavailable — fail silently, log still works in-memory
    }
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  /* ----------------------------------------------------------
     Init
  ---------------------------------------------------------- */
  renderLog();
  renderStats();

})();
