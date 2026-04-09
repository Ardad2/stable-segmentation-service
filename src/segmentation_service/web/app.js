"use strict";

// ---------------------------------------------------------------------------
// API Details helpers
// ---------------------------------------------------------------------------

// Fields whose values are truncated for display (base64 blobs).
const _BLOB_FIELDS = new Set(["image", "mask_b64", "mask_data", "logits_b64"]);

/**
 * Return a deep copy of `val` with long strings shortened for display.
 * Only shortens known blob fields OR any string > 200 chars.
 * Never mutates the original.
 */
function truncateLongFields(val, key = null) {
  if (typeof val === "string") {
    const isBlobField = key !== null && _BLOB_FIELDS.has(key);
    if (isBlobField || val.length > 200) {
      const keep = 80;
      if (val.length > keep) {
        return val.slice(0, keep) + `…[${val.length} chars]`;
      }
    }
    return val;
  }
  if (Array.isArray(val)) {
    return val.map((item) => truncateLongFields(item));
  }
  if (val !== null && typeof val === "object") {
    const out = {};
    for (const [k, v] of Object.entries(val)) {
      out[k] = truncateLongFields(v, k);
    }
    return out;
  }
  return val;
}

/**
 * Update the API Details panel.
 * @param {object} opts
 * @param {string}      opts.method        HTTP method (GET / POST)
 * @param {string}      opts.endpoint      Path, e.g. "/api/v1/segment"
 * @param {number|null} [opts.status]      HTTP status code, or null while pending
 * @param {object|null} [opts.requestBody] Payload sent (null for GET)
 * @param {object|null} [opts.responseBody] Parsed response (null while pending)
 */
function setApiDetails({ method, endpoint, status = null, requestBody = null, responseBody = null }) {
  // Activity line
  const activityEl = document.getElementById("api-activity");
  let activity = `${method}  ${endpoint}`;
  if (status !== null) {
    const ok = status >= 200 && status < 300;
    activity += `  →  ${status}`;
    activityEl.className = ok ? "api-status-ok" : "api-status-err";
  } else {
    activityEl.className = "api-status-pending";
  }
  activityEl.textContent = activity;

  // Request block
  const reqPre = document.getElementById("api-req-pre");
  if (requestBody === null) {
    reqPre.textContent = "(no request body)";
    reqPre.classList.add("api-placeholder");
  } else {
    reqPre.textContent = JSON.stringify(truncateLongFields(requestBody), null, 2);
    reqPre.classList.remove("api-placeholder");
  }

  // Response block
  const resPre = document.getElementById("api-res-pre");
  if (responseBody === null) {
    resPre.textContent = "—";
    resPre.classList.add("api-placeholder");
  } else {
    resPre.textContent = JSON.stringify(truncateLongFields(responseBody), null, 2);
    resPre.classList.remove("api-placeholder");
  }

  // Auto-expand on first real activity
  document.getElementById("api-details").open = true;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let imageB64 = null;       // base64 string (no data-url prefix)
let imageFormat = "png";
let pointCoords = null;    // {x, y} in image pixels

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const imageInput      = document.getElementById("image-input");
const imageCanvas     = document.getElementById("image-canvas");
const maskCanvas      = document.getElementById("mask-canvas");
const placeholder     = document.getElementById("canvas-placeholder");
const pointDisplay    = document.getElementById("point-display");
const textInput       = document.getElementById("text-prompt-input");
const submitBtn       = document.getElementById("submit-btn");
const errorMsg        = document.getElementById("error-msg");
const resultPanel     = document.getElementById("result-panel");
const textSection     = document.getElementById("text-prompt-section");
const pointSection    = document.getElementById("point-prompt-section");
const backendLabel    = document.getElementById("backend-label");
const healthLabel     = document.getElementById("health-label");

const imgCtx  = imageCanvas.getContext("2d");
const maskCtx = maskCanvas.getContext("2d");

// ---------------------------------------------------------------------------
// Init: fetch health + capabilities
// ---------------------------------------------------------------------------

async function init() {
  try {
    const [healthRes, capsRes] = await Promise.all([
      fetch("/api/v1/health"),
      fetch("/api/v1/capabilities"),
    ]);

    if (healthRes.ok) {
      const h = await healthRes.json();
      healthLabel.textContent = `status: ${h.status}`;
      healthLabel.classList.add("ok");
      setApiDetails({ method: "GET", endpoint: "/api/v1/health", status: healthRes.status, requestBody: null, responseBody: h });
    } else {
      healthLabel.textContent = "status: error";
      healthLabel.classList.add("err");
      setApiDetails({ method: "GET", endpoint: "/api/v1/health", status: healthRes.status, requestBody: null, responseBody: null });
    }

    if (capsRes.ok) {
      const c = await capsRes.json();
      backendLabel.textContent = `backend: ${c.backend}`;
      setApiDetails({ method: "GET", endpoint: "/api/v1/capabilities", status: capsRes.status, requestBody: null, responseBody: c });

      // Disable prompt types not supported by this backend.
      const supported = new Set(c.supported_prompt_types || []);
      for (const radio of document.querySelectorAll('input[name="prompt_type"]')) {
        if (!supported.has(radio.value)) {
          radio.disabled = true;
          radio.parentElement.title = `Not supported by ${c.backend}`;
          radio.parentElement.style.opacity = "0.4";
        }
      }
      // Select first enabled radio automatically.
      const firstEnabled = document.querySelector('input[name="prompt_type"]:not(:disabled)');
      if (firstEnabled) {
        firstEnabled.checked = true;
        onPromptTypeChange(firstEnabled.value);
      }
    }
  } catch (e) {
    healthLabel.textContent = "status: unreachable";
    healthLabel.classList.add("err");
    setApiDetails({ method: "GET", endpoint: "/api/v1/health", status: null, requestBody: null, responseBody: { error: e.message } });
  }
}

// ---------------------------------------------------------------------------
// Image upload
// ---------------------------------------------------------------------------

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;

  const ext = file.name.split(".").pop().toLowerCase();
  imageFormat = (ext === "jpg" || ext === "jpeg") ? "jpeg" : "png";

  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    // Strip the data-url prefix to get bare base64.
    imageB64 = dataUrl.split(",")[1];

    // Draw to canvas.
    const img = new Image();
    img.onload = () => {
      imageCanvas.width  = img.naturalWidth;
      imageCanvas.height = img.naturalHeight;
      maskCanvas.width   = img.naturalWidth;
      maskCanvas.height  = img.naturalHeight;
      imgCtx.drawImage(img, 0, 0);
      placeholder.style.display = "none";
      pointCoords = null;
      pointDisplay.textContent = "Point: none";
      clearMask();
      updateSubmitState();
    };
    img.src = dataUrl;
  };
  reader.readAsDataURL(file);
});

// ---------------------------------------------------------------------------
// Prompt type toggle
// ---------------------------------------------------------------------------

document.querySelectorAll('input[name="prompt_type"]').forEach((radio) => {
  radio.addEventListener("change", () => onPromptTypeChange(radio.value));
});

function onPromptTypeChange(value) {
  textSection.classList.toggle("hidden", value !== "text");
  pointSection.classList.toggle("hidden", value !== "point");
  if (value !== "point") pointCoords = null;
  updateSubmitState();
}

// ---------------------------------------------------------------------------
// Point selection via canvas click
// ---------------------------------------------------------------------------

imageCanvas.addEventListener("click", (e) => {
  const promptType = getPromptType();
  if (promptType !== "point" || !imageB64) return;

  const rect = imageCanvas.getBoundingClientRect();
  const scaleX = imageCanvas.width  / rect.width;
  const scaleY = imageCanvas.height / rect.height;
  pointCoords = {
    x: Math.round((e.clientX - rect.left) * scaleX),
    y: Math.round((e.clientY - rect.top)  * scaleY),
  };
  pointDisplay.textContent = `Point: (${pointCoords.x}, ${pointCoords.y})`;

  // Draw a small crosshair marker.
  clearMask();
  maskCtx.save();
  maskCtx.strokeStyle = "#ff4444";
  maskCtx.lineWidth = Math.max(1, imageCanvas.width / 200);
  const r = Math.max(6, imageCanvas.width / 60);
  maskCtx.beginPath();
  maskCtx.arc(pointCoords.x, pointCoords.y, r, 0, Math.PI * 2);
  maskCtx.stroke();
  maskCtx.beginPath();
  maskCtx.moveTo(pointCoords.x - r * 1.5, pointCoords.y);
  maskCtx.lineTo(pointCoords.x + r * 1.5, pointCoords.y);
  maskCtx.stroke();
  maskCtx.beginPath();
  maskCtx.moveTo(pointCoords.x, pointCoords.y - r * 1.5);
  maskCtx.lineTo(pointCoords.x, pointCoords.y + r * 1.5);
  maskCtx.stroke();
  maskCtx.restore();

  updateSubmitState();
});

// ---------------------------------------------------------------------------
// Submit
// ---------------------------------------------------------------------------

submitBtn.addEventListener("click", runSegmentation);

async function runSegmentation() {
  clearError();
  clearMask();
  submitBtn.disabled = true;
  submitBtn.textContent = "Running…";

  const promptType = getPromptType();
  const payload = {
    image: imageB64,
    image_format: imageFormat,
    prompt_type: promptType,
  };

  if (promptType === "text") {
    payload.text_prompt = textInput.value.trim();
  } else if (promptType === "point" && pointCoords) {
    payload.points = [{ x: pointCoords.x, y: pointCoords.y, label: 1 }];
  }

  // Show the outgoing request immediately (pending, no response yet).
  setApiDetails({ method: "POST", endpoint: "/api/v1/segment", status: null, requestBody: payload, responseBody: null });

  try {
    const res = await fetch("/api/v1/segment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    setApiDetails({ method: "POST", endpoint: "/api/v1/segment", status: res.status, requestBody: payload, responseBody: data });

    if (!res.ok) {
      showError(`Server error ${res.status}: ${data.detail || JSON.stringify(data)}`);
      return;
    }

    displayResult(data);
  } catch (e) {
    showError(`Request failed: ${e.message}`);
    setApiDetails({ method: "POST", endpoint: "/api/v1/segment", status: null, requestBody: payload, responseBody: { error: e.message } });
  } finally {
    submitBtn.textContent = "Run segmentation";
    updateSubmitState();
  }
}

// ---------------------------------------------------------------------------
// Result display
// ---------------------------------------------------------------------------

function displayResult(data) {
  const masks = data.masks || [];
  document.getElementById("res-backend").textContent  = data.backend || "—";
  document.getElementById("res-latency").textContent  = data.latency_ms != null ? `${data.latency_ms.toFixed(1)} ms` : "—";
  document.getElementById("res-masks").textContent    = masks.length;
  document.getElementById("res-score").textContent    = masks[0]?.score  != null ? masks[0].score.toFixed(4)  : "—";
  document.getElementById("res-area").textContent     = masks[0]?.area   != null ? masks[0].area              : "—";
  resultPanel.classList.remove("hidden");

  if (masks.length === 0) return;

  // Overlay the top mask (semi-transparent teal fill).
  const maskB64 = masks[0].mask_b64;
  if (!maskB64) return;

  const img = new Image();
  img.onload = () => {
    // Draw mask into an offscreen canvas to read pixels.
    const off = document.createElement("canvas");
    off.width  = img.naturalWidth;
    off.height = img.naturalHeight;
    const offCtx = off.getContext("2d");
    offCtx.drawImage(img, 0, 0);

    const src  = offCtx.getImageData(0, 0, off.width, off.height);
    const dest = maskCtx.createImageData(imageCanvas.width, imageCanvas.height);

    // Scale mask to canvas if sizes differ.
    const sw = imageCanvas.width  / off.width;
    const sh = imageCanvas.height / off.height;

    for (let dy = 0; dy < imageCanvas.height; dy++) {
      for (let dx = 0; dx < imageCanvas.width; dx++) {
        const sx = Math.min(Math.floor(dx / sw), off.width  - 1);
        const sy = Math.min(Math.floor(dy / sh), off.height - 1);
        const si = (sy * off.width + sx) * 4;
        const di = (dy * imageCanvas.width + dx) * 4;
        // Non-black pixel → masked region.
        if (src.data[si] > 0 || src.data[si+1] > 0 || src.data[si+2] > 0) {
          dest.data[di]   = 0;
          dest.data[di+1] = 200;
          dest.data[di+2] = 180;
          dest.data[di+3] = 110;  // ~43% opacity
        }
      }
    }
    maskCtx.putImageData(dest, 0, 0);
  };
  img.src = "data:image/png;base64," + maskB64;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getPromptType() {
  const checked = document.querySelector('input[name="prompt_type"]:checked');
  return checked ? checked.value : "text";
}

function updateSubmitState() {
  if (!imageB64) { submitBtn.disabled = true; return; }
  const pt = getPromptType();
  if (pt === "point" && !pointCoords) { submitBtn.disabled = true; return; }
  if (pt === "text"  && !textInput.value.trim()) { submitBtn.disabled = true; return; }
  if (pt === "box")  { submitBtn.disabled = true; return; }
  submitBtn.disabled = false;
}

textInput.addEventListener("input", updateSubmitState);

function clearMask() {
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.remove("hidden");
}

function clearError() {
  errorMsg.textContent = "";
  errorMsg.classList.add("hidden");
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

init();
