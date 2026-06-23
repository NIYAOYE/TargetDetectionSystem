const state = {
  config: null,
  files: null,
  image: null,
  imageName: "",
  imageUrl: "",
  imageWidth: 0,
  imageHeight: 0,
  imageLoaded: false,
  imageLoadToken: 0,
  detections: [],
  detectionsImageName: "",
  selectedDetection: -1,
  currentJobId: "",
  currentJobImageName: "",
  pollTimer: null,
  mode: "normal",
  view: { scale: 1, offsetX: 0, offsetY: 0 },
  dragging: false,
  panning: false,
  drawStart: null,
  drawCurrent: null,
  panStart: null,
  leftResize: { dragging: false, startY: 0, startTop: 0 },
};

const els = {};

document.addEventListener("DOMContentLoaded", () => {
  for (const id of [
    "detectorSelect",
    "classifierSelect",
    "refreshFilesBtn",
    "imageSelect",
    "patchSizeInput",
    "overlapInfo",
    "useRfCheckbox",
    "runDetectionBtn",
    "statusBox",
    "addModeBtn",
    "deleteModeBtn",
    "exportBtn",
    "exportLinks",
    "resultsTable",
    "imageCanvas",
    "imageTitle",
    "zoomLabel",
    "globalStatus",
    "jobProgress",
    "leftResizeHandle",
    "headerDot",
    "statusDot",
    "segmentPanel",
    "segmentImage",
    "segmentCloseBtn",
  ]) {
    els[id] = document.getElementById(id);
  }

  bindEvents();
  init();
});

function bindEvents() {
  els.refreshFilesBtn.addEventListener("click", refreshFiles);
  els.imageSelect.addEventListener("change", loadSelectedImage);
  els.imageSelect.addEventListener("click", () => {
    // Handle single-option select: click doesn't fire change when already selected
    const imageName = ensureSelectValue(els.imageSelect);
    if (imageName && (!state.image || state.imageName !== imageName)) {
      loadSelectedImage();
    }
  });
  els.runDetectionBtn.addEventListener("click", runDetection);
  els.addModeBtn.addEventListener("click", () => toggleMode("draw"));
  els.deleteModeBtn.addEventListener("click", () => toggleMode("delete"));
  els.exportBtn.addEventListener("click", exportResults);
  bindLeftResize();
  bindFileActions();

  if (els.segmentCloseBtn) {
    els.segmentCloseBtn.addEventListener("click", hideSegmentPanel);
  }
  if (els.segmentImage) {
    els.segmentImage.addEventListener("click", () => openSegment(state.selectedDetection));
  }

  const canvas = els.imageCanvas;
  canvas.addEventListener("wheel", onCanvasWheel, { passive: false });
  canvas.addEventListener("mousedown", onCanvasMouseDown);
  canvas.addEventListener("mousemove", onCanvasMouseMove);
  canvas.addEventListener("mouseup", onCanvasMouseUp);
  canvas.addEventListener("mouseleave", onCanvasMouseLeave);
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());

  window.addEventListener("resize", () => {
    initializeLeftSplit(false);
    resizeCanvas();
    drawCanvas();
  });
}

async function init() {
  await loadConfig();
  await refreshFiles();
  initializeLeftSplit(false);
  resizeCanvas();
  drawCanvas();
  document.body.classList.remove("is-processing");
  setStatus("就绪");
}

function bindLeftResize() {
  if (!els.leftResizeHandle) return;

  els.leftResizeHandle.addEventListener("mousedown", (event) => {
    event.preventDefault();
    const sideColumn = document.querySelector(".side-column");
    const controlsStack = document.querySelector(".controls-stack");
    if (!sideColumn || !controlsStack) return;

    state.leftResize.dragging = true;
    state.leftResize.startY = event.clientY;
    state.leftResize.startTop = controlsStack.getBoundingClientRect().height;
    document.body.classList.add("left-resizing");
  });

  els.leftResizeHandle.addEventListener("keydown", (event) => {
    if (!["ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const currentTop = document.querySelector(".controls-stack")?.getBoundingClientRect().height || 0;
    if (event.key === "ArrowUp") setLeftSplitHeight(currentTop - 32);
    if (event.key === "ArrowDown") setLeftSplitHeight(currentTop + 32);
    if (event.key === "Home") setLeftSplitHeight(260);
    if (event.key === "End") setLeftSplitHeight(Number.MAX_SAFE_INTEGER);
  });

  window.addEventListener("mousemove", (event) => {
    if (!state.leftResize.dragging) return;
    setLeftSplitHeight(state.leftResize.startTop + event.clientY - state.leftResize.startY);
  });

  window.addEventListener("mouseup", () => {
    if (!state.leftResize.dragging) return;
    state.leftResize.dragging = false;
    document.body.classList.remove("left-resizing");
  });
}

function initializeLeftSplit(useDefault) {
  const sideColumn = document.querySelector(".side-column");
  if (!sideColumn || !els.leftResizeHandle) return;

  const metrics = getLeftSplitMetrics();
  if (!metrics) return;

  const saved = Number(window.localStorage.getItem("sarLeftTopHeight"));
  const defaultHeight = Math.round(metrics.available * 0.64);
  const topHeight = !useDefault && Number.isFinite(saved) && saved > 0 ? saved : defaultHeight;
  setLeftSplitHeight(topHeight, false);
}

function getLeftSplitMetrics() {
  const sideColumn = document.querySelector(".side-column");
  if (!sideColumn || !els.leftResizeHandle) return null;

  const styles = window.getComputedStyle(sideColumn);
  const rowGap = Number.parseFloat(styles.rowGap) || 0;
  const handleHeight = els.leftResizeHandle.getBoundingClientRect().height || 10;
  const available = sideColumn.clientHeight - handleHeight - rowGap * 2;
  if (available <= 0) return null;

  const minTop = Math.min(260, Math.max(180, Math.round(available * 0.28)));
  const minBottom = Math.min(220, Math.max(150, Math.round(available * 0.22)));
  const maxTop = Math.max(minTop, available - minBottom);
  return { available, handleHeight, minTop, minBottom, maxTop };
}

function setLeftSplitHeight(topHeight, persist = true) {
  const sideColumn = document.querySelector(".side-column");
  const metrics = getLeftSplitMetrics();
  if (!sideColumn || !metrics) return;

  const clamped = Math.max(metrics.minTop, Math.min(metrics.maxTop, topHeight));
  sideColumn.style.gridTemplateRows = `${Math.round(clamped)}px ${metrics.handleHeight}px minmax(${Math.round(metrics.minBottom)}px, 1fr)`;
  if (persist) {
    window.localStorage.setItem("sarLeftTopHeight", String(Math.round(clamped)));
  }
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) {
      // Keep the HTTP status as the visible error.
    }
    throw new Error(detail);
  }
  return response.json();
}

async function loadConfig() {
  state.config = await apiFetch("/api/config");
  els.patchSizeInput.value = state.config.patch_size;
  els.overlapInfo.textContent = `切片重叠固定为 ${state.config.overlap} px`;
  els.useRfCheckbox.checked = Boolean(state.config.default_rf_enabled);
}

async function refreshFiles() {
  try {
    const previousImageName = state.imageName || els.imageSelect.value;
    state.files = await apiFetch("/api/files");
    fillSelect(els.detectorSelect, state.files.detectors, "未发现检测模型");
    const selectedImageName = fillSelect(els.imageSelect, state.files.images, "未发现 SAR 图像", previousImageName);
    fillClassifierSelect();

    // Keep state in sync even when the select only has one option and no change event fires.
    if (selectedImageName) {
      loadSelectedImage();
    } else {
      updateRunButton();
    }

    setStatus("文件列表已刷新");
  } catch (error) {
    setStatus(`文件列表刷新失败: ${error.message}`);
  }
}

function fillSelect(select, files, emptyText, preferredValue = select.value) {
  select.innerHTML = "";
  if (!files.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = emptyText;
    option.selected = true;
    select.appendChild(option);
    return "";
  }

  const selectedValue = files.some((file) => file.name === preferredValue) ? preferredValue : files[0].name;
  for (const file of files) {
    const option = document.createElement("option");
    option.value = file.name;
    option.textContent = file.display_name;
    option.selected = file.name === selectedValue;
    select.appendChild(option);
  }
  select.value = selectedValue;
  return selectedValue;
}

function fillClassifierSelect() {
  els.classifierSelect.innerHTML = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "无 RF 分类模型";
  els.classifierSelect.appendChild(none);

  for (const file of state.files.classifiers) {
    const option = document.createElement("option");
    option.value = file.name;
    option.textContent = file.display_name;
    els.classifierSelect.appendChild(option);
  }
}

function updateRunButton() {
  const imageName = ensureSelectValue(els.imageSelect);
  const detectorName = ensureSelectValue(els.detectorSelect);
  els.runDetectionBtn.disabled = !imageName || !detectorName || Boolean(state.pollTimer);
}

function setStatus(message) {
  els.globalStatus.textContent = message;
  updateStatusDots(message);
}

function updateStatusDots(message) {
  const text = message.toLowerCase();
  els.headerDot.className = "header-status-dot";
  els.statusDot.className = "status-dot";
  if (text.includes("失败") || text.includes("错误") || text.includes("error") || text.includes("fail")) {
    els.headerDot.classList.add("error");
    els.statusDot.classList.add("error");
  } else if (text.includes("运行") || text.includes("检测中") || text.includes("processing") || text.includes("running")) {
    els.headerDot.classList.add("warning");
    els.statusDot.classList.add("warning");
  }
}

function setStatusBox(lines) {
  els.statusBox.value = lines.filter(Boolean).join("\n");
}

function ensureSelectValue(select) {
  if (!select.value) {
    const selectableIndex = Array.from(select.options).findIndex((option) => option.value);
    if (selectableIndex !== -1) {
      select.selectedIndex = selectableIndex;
    }
  }
  return select.value;
}

function revokeImageUrl() {
  if (state.imageUrl) {
    URL.revokeObjectURL(state.imageUrl);
    state.imageUrl = "";
  }
}

async function loadSelectedImage() {
  const imageName = ensureSelectValue(els.imageSelect);
  const loadToken = state.imageLoadToken + 1;
  state.imageLoadToken = loadToken;

  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  document.body.classList.remove("is-processing");
  els.jobProgress.hidden = true;

  state.imageName = imageName;
  state.image = null;
  revokeImageUrl();
  const imageInfo = state.files?.images?.find((file) => file.name === imageName);
  state.imageWidth = Number(imageInfo?.width) || 0;
  state.imageHeight = Number(imageInfo?.height) || 0;
  state.imageLoaded = false;
  state.currentJobId = "";
  state.currentJobImageName = "";
  state.detections = [];
  state.detectionsImageName = "";
  state.selectedDetection = -1;
  state.mode = "normal";
  hideSegmentPanel();
  updateModeButtons();
  els.exportBtn.disabled = true;
  els.addModeBtn.disabled = true;
  els.deleteModeBtn.disabled = true;
  els.exportLinks.innerHTML = "";
  renderTable();

  if (!imageName) {
    state.image = null;
    els.imageTitle.textContent = "未加载图像";
    drawCanvas();
    updateRunButton();
    return;
  }

  els.imageTitle.textContent = imageName;
  setStatus(`Loading image preview: ${imageName}`);
  updateRunButton();
  drawCanvas();

  const img = new Image();
  img.dataset.fullWidth = String(state.imageWidth || 0);
  img.dataset.fullHeight = String(state.imageHeight || 0);
  img.onload = () => {
    if (loadToken !== state.imageLoadToken || state.imageName !== imageName || els.imageSelect.value !== imageName) {
      return;
    }
    state.image = img;
    state.imageLoaded = true;
    state.imageWidth = Number(img.dataset.fullWidth) || state.imageWidth || img.width;
    state.imageHeight = Number(img.dataset.fullHeight) || state.imageHeight || img.height;
    fitImageToCanvas();
    const shouldUpdatePreviewStatus = !state.currentJobId;
    els.imageTitle.textContent = imageName;
    if (shouldUpdatePreviewStatus) {
    setStatus(`已加载图像: ${imageName}`);
    setStatusBox([
      `当前图像：${imageName}`,
      `当前切片大小：${els.patchSizeInput.value} px`,
      "检测状态：等待运行检测",
    ]);
    }
    updateRunButton();
    drawCanvas();
  };
  img.onerror = () => {
    if (loadToken !== state.imageLoadToken || state.imageName !== imageName || els.imageSelect.value !== imageName) {
      return;
    }
    state.image = null;
    state.imageLoaded = false;
    state.imageWidth = 0;
    state.imageHeight = 0;
    setStatus("图像预览加载失败");
    updateRunButton();
    drawCanvas();
  };
  img.src = `/api/images/${encodeURIComponent(imageName)}/preview.png?ts=${Date.now()}`;
}

async function runDetection() {
  // Fallback: if imageName wasn't set via change event, use select value
  const selectedImageName = ensureSelectValue(els.imageSelect);
  if (selectedImageName && state.imageName !== selectedImageName) {
    loadSelectedImage();
    setStatus("Waiting for image preview to load...");
    return;
  }
  if (!state.imageName) {
    setStatus("请先选择 SAR 图像");
    return;
  }

  const jobImageName = state.imageName;

  const request = {
    image_name: jobImageName,
    detector_name: els.detectorSelect.value,
    classifier_name: els.classifierSelect.value || null,
    patch_size: Number(els.patchSizeInput.value),
    use_classifier: els.useRfCheckbox.checked,
    conf_threshold: state.config.conf_threshold,
    iou_threshold: state.config.iou_threshold,
    nms_threshold: state.config.nms_threshold,
    device: state.config.device,
  };

  try {
    document.body.classList.add("is-processing");
    els.runDetectionBtn.disabled = true;
    els.exportBtn.disabled = true;
    els.exportLinks.innerHTML = "";
    els.jobProgress.hidden = false;
    els.jobProgress.value = 0;
    state.detections = [];
    state.detectionsImageName = "";
    state.selectedDetection = -1;
    renderTable();
    hideSegmentPanel();
    drawCanvas();

    const job = await apiFetch("/api/jobs/detect", {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (state.imageName !== jobImageName) {
      document.body.classList.remove("is-processing");
      els.jobProgress.hidden = true;
      updateRunButton();
      return;
    }
    state.currentJobId = job.job_id;
    state.currentJobImageName = jobImageName;
    setStatus("检测任务已提交");
    setStatusBox([
      `当前图像：${state.imageName}`,
      `当前切片大小：${request.patch_size} px`,
      `RF 分类：${request.use_classifier && request.classifier_name ? "启用" : "未启用"}`,
      "检测状态：正在运行，请稍候",
    ]);
    startPolling();
  } catch (error) {
    els.jobProgress.hidden = true;
    updateRunButton();
    setStatus(`检测启动失败: ${error.message}`);
  }
}

function startPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
  }
  state.pollTimer = setInterval(pollJob, 700);
  pollJob();
}

async function pollJob() {
  const jobId = state.currentJobId;
  const jobImageName = state.currentJobImageName;
  if (!jobId) {
    return;
  }

  try {
    const job = await apiFetch(`/api/jobs/${jobId}`);
    if (jobId !== state.currentJobId || jobImageName !== state.currentJobImageName || job.image_name !== state.imageName) {
      return;
    }
    els.jobProgress.value = job.progress_current;
    setStatus(job.message);
    setStatusBox([
      `当前图像：${job.image_name}`,
      `当前切片大小：${els.patchSizeInput.value} px`,
      `检测状态：${job.message}`,
    ]);

    if (job.status === "completed") {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      document.body.classList.remove("is-processing");
      els.jobProgress.hidden = true;
      els.addModeBtn.disabled = false;
      els.deleteModeBtn.disabled = false;
      await refreshDetections(jobId, jobImageName);
      els.exportBtn.disabled = state.detections.length === 0;
      updateRunButton();
      setStatus(`检测完成，发现 ${state.detections.length} 个目标`);
    } else if (job.status === "error") {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      document.body.classList.remove("is-processing");
      els.jobProgress.hidden = true;
      updateRunButton();
      setStatus(`检测失败: ${job.error || job.message}`);
      setStatusBox([
        `当前图像：${job.image_name}`,
        `当前切片大小：${els.patchSizeInput.value} px`,
        "检测状态：运行失败",
        `错误信息：${job.error || job.message}`,
      ]);
    }
  } catch (error) {
    document.body.classList.remove("is-processing");
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    els.jobProgress.hidden = true;
    updateRunButton();
    setStatus(`任务查询失败: ${error.message}`);
  }
}

async function refreshDetections(jobId = state.currentJobId, imageName = state.currentJobImageName) {
  if (!jobId) {
    state.detections = [];
    state.detectionsImageName = "";
  } else {
    const detections = await apiFetch(`/api/jobs/${jobId}/detections`);
    if (jobId !== state.currentJobId || imageName !== state.currentJobImageName || imageName !== state.imageName) {
      return false;
    }
    state.detections = detections;
    state.detectionsImageName = imageName;
  }
  state.selectedDetection = -1;
  renderTable();
  hideSegmentPanel();
  drawCanvas();
  return true;
}

function renderTable() {
  const tbody = els.resultsTable.querySelector("tbody");
  tbody.innerHTML = "";

  if (!state.detections.length) {
    const row = document.createElement("tr");
    row.className = "empty-row";
    const cell = document.createElement("td");
    cell.colSpan = 12;
    cell.textContent = state.currentJobId ? "暂无检测结果，任务完成后会在这里显示目标明细。" : "选择 SAR 图像并运行检测后，结果会出现在这里。";
    row.appendChild(cell);
    tbody.appendChild(row);
    return;
  }

  for (const detection of state.detections) {
    const row = document.createElement("tr");
    row.dataset.id = detection.id;
    if (detection.id === state.selectedDetection) {
      row.classList.add("selected");
    }

    const values = [
      detection.id,
      detection.class_name,
      detection.confidence.toFixed(3),
      Math.round(detection.bbox[0]),
      Math.round(detection.bbox[1]),
      Math.round(detection.bbox[2]),
      Math.round(detection.bbox[3]),
      detection.scatter_point_count,
      detection.scatter_mean_amplitude.toFixed(4),
      detection.hu_moment_1.toFixed(4),
      detection.hu_moment_2.toFixed(4),
      rfText(detection),
    ];

    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index === values.length - 1) {
        cell.className = rfClass(detection);
      }
      row.appendChild(cell);
    });

    row.title = "点击查看目标抠图";
    row.addEventListener("click", () => selectDetection(detection.id, true));
    tbody.appendChild(row);
  }
}

function segmentUrl(id) {
  return `/api/jobs/${state.currentJobId}/detections/${id}/segment.png?ts=${Date.now()}`;
}

function updateSegmentPanel() {
  const panel = els.segmentPanel;
  const img = els.segmentImage;
  if (!panel || !img) return;

  const id = state.selectedDetection;
  const exists = state.currentJobId && id >= 0 && state.detections.some((d) => d.id === id);
  if (!exists) {
    hideSegmentPanel();
    return;
  }
  img.src = segmentUrl(id);
  panel.hidden = false;
}

function hideSegmentPanel() {
  if (!els.segmentPanel) return;
  els.segmentPanel.hidden = true;
  if (els.segmentImage) els.segmentImage.removeAttribute("src");
}

function openSegment(id) {
  if (!state.currentJobId || id < 0) return;
  window.open(segmentUrl(id), "_blank", "noreferrer");
}

function rfText(detection) {
  if (detection.source === "manual") return "手动添加";
  if (detection.rf_result === 1) return "真目标";
  if (detection.rf_result === 0) return "虚警";
  return "未分类";
}

function rfClass(detection) {
  if (detection.source === "manual") return "rf-manual";
  if (detection.rf_result === 1) return "rf-true";
  if (detection.rf_result === 0) return "rf-false";
  return "rf-unknown";
}

function detectionColor(detection) {
  if (detection.source === "manual") return "#65a7ff";
  if (detection.rf_result === 1) return "#51e88c";
  if (detection.rf_result === 0) return "#ff6673";
  return "#f4d35e";
}

function selectDetection(id, center) {
  state.selectedDetection = id;
  renderTable();
  updateSegmentPanel();
  if (center) {
    const detection = state.detections[id];
    if (detection) {
      const [x, y, w, h] = detection.bbox;
      const cx = x + w / 2;
      const cy = y + h / 2;
      state.view.offsetX = els.imageCanvas.width / 2 - cx * state.view.scale;
      state.view.offsetY = els.imageCanvas.height / 2 - cy * state.view.scale;
    }
  }
  drawCanvas();
}

function toggleMode(mode) {
  state.mode = state.mode === mode ? "normal" : mode;
  updateModeButtons();
  drawCanvas();
}

function updateModeButtons() {
  els.addModeBtn.classList.toggle("active", state.mode === "draw");
  els.deleteModeBtn.classList.toggle("active", state.mode === "delete");
}

function resizeCanvas() {
  const canvas = els.imageCanvas;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width));
  canvas.height = Math.max(1, Math.floor(rect.height));
  if (state.image) {
    fitImageToCanvas(false);
  }
}

function fitImageToCanvas(resetZoom = true) {
  if (!state.image) return;
  const canvas = els.imageCanvas;
  const imageWidth = state.imageWidth || state.image.width;
  const imageHeight = state.imageHeight || state.image.height;
  const scale = Math.min(canvas.width / imageWidth, canvas.height / imageHeight) * 0.96;
  if (resetZoom || !Number.isFinite(state.view.scale)) {
    state.view.scale = Math.max(scale, 0.01);
  }
  state.view.offsetX = (canvas.width - imageWidth * state.view.scale) / 2;
  state.view.offsetY = (canvas.height - imageHeight * state.view.scale) / 2;
  updateZoomLabel();
}

function drawCanvas() {
  const canvas = els.imageCanvas;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#0a0f16";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (!state.image) {
    ctx.fillStyle = "#9cb2b2";
    ctx.font = "14px Microsoft YaHei, Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("未加载图像", canvas.width / 2, canvas.height / 2);
    return;
  }

  ctx.save();
  ctx.translate(state.view.offsetX, state.view.offsetY);
  ctx.scale(state.view.scale, state.view.scale);
  ctx.drawImage(state.image, 0, 0, state.imageWidth || state.image.width, state.imageHeight || state.image.height);

  if (state.detectionsImageName === state.imageName) {
    for (const detection of state.detections) {
      drawDetection(ctx, detection, detection.id === state.selectedDetection);
    }
  }

  if (state.drawStart && state.drawCurrent) {
    const rect = normalizedRect(state.drawStart, state.drawCurrent);
    ctx.strokeStyle = "#65a7ff";
    ctx.lineWidth = 2 / state.view.scale;
    ctx.setLineDash([8 / state.view.scale, 5 / state.view.scale]);
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
    ctx.setLineDash([]);
  }

  ctx.restore();
  updateZoomLabel();
}

function drawDetection(ctx, detection, selected) {
  const [x, y, w, h] = detection.bbox;
  const color = detectionColor(detection);
  ctx.strokeStyle = color;
  ctx.lineWidth = (selected ? 4 : 2) / state.view.scale;
  ctx.strokeRect(x, y, w, h);

  const label = `${detection.class_name} ${detection.confidence.toFixed(2)}`;
  const fontSize = 13 / state.view.scale;
  ctx.font = `${fontSize}px Microsoft YaHei, Segoe UI, sans-serif`;
  const metrics = ctx.measureText(label);
  const labelHeight = 18 / state.view.scale;
  const labelWidth = metrics.width + 8 / state.view.scale;
  const labelY = Math.max(0, y - labelHeight);
  ctx.fillStyle = "rgba(4, 12, 18, 0.86)";
  ctx.fillRect(x, labelY, labelWidth, labelHeight);
  ctx.fillStyle = color;
  ctx.fillText(label, x + 4 / state.view.scale, labelY + 13 / state.view.scale);
}

function updateZoomLabel() {
  els.zoomLabel.textContent = `${Math.round(state.view.scale * 100)}%`;
}

function onCanvasWheel(event) {
  if (!state.image) return;
  event.preventDefault();
  const canvasPos = eventToCanvas(event);
  const imagePos = canvasToImage(canvasPos);
  const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
  const nextScale = Math.max(0.02, Math.min(20, state.view.scale * factor));
  state.view.scale = nextScale;
  state.view.offsetX = canvasPos.x - imagePos.x * nextScale;
  state.view.offsetY = canvasPos.y - imagePos.y * nextScale;
  drawCanvas();
}

function onCanvasMouseDown(event) {
  if (!state.image) return;
  const canvasPos = eventToCanvas(event);
  const imagePos = canvasToImage(canvasPos);

  if (event.button === 1 || event.button === 2) {
    state.panning = true;
    state.panStart = { x: event.clientX, y: event.clientY, ox: state.view.offsetX, oy: state.view.offsetY };
    return;
  }

  if (event.button !== 0) return;

  if (state.mode === "draw") {
    state.drawStart = clampImagePoint(imagePos);
    state.drawCurrent = state.drawStart;
    state.dragging = true;
    drawCanvas();
    return;
  }

  const hit = hitTest(imagePos);
  if (state.mode === "delete") {
    if (hit >= 0) deleteDetection(hit);
    return;
  }

  if (hit >= 0) {
    selectDetection(hit, false);
  } else {
    state.selectedDetection = -1;
    renderTable();
    hideSegmentPanel();
    drawCanvas();
  }
}

function onCanvasMouseMove(event) {
  if (!state.image) return;

  if (state.panning && state.panStart) {
    state.view.offsetX = state.panStart.ox + event.clientX - state.panStart.x;
    state.view.offsetY = state.panStart.oy + event.clientY - state.panStart.y;
    drawCanvas();
    return;
  }

  if (state.dragging && state.mode === "draw") {
    state.drawCurrent = clampImagePoint(canvasToImage(eventToCanvas(event)));
    drawCanvas();
  }
}

async function onCanvasMouseUp(event) {
  if (event.button === 1 || event.button === 2) {
    state.panning = false;
    state.panStart = null;
    return;
  }

  if (!state.dragging || state.mode !== "draw") return;
  state.dragging = false;
  const rect = normalizedRect(state.drawStart, state.drawCurrent);
  state.drawStart = null;
  state.drawCurrent = null;
  drawCanvas();

  if (rect.w < 5 || rect.h < 5) return;
  const className = window.prompt("请输入目标类别名称", "manual_target");
  if (!className || !className.trim()) return;
  await addManualDetection([rect.x, rect.y, rect.w, rect.h], className.trim());
}

function onCanvasMouseLeave() {
  state.panning = false;
  state.panStart = null;
}

function eventToCanvas(event) {
  const rect = els.imageCanvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

function canvasToImage(point) {
  return {
    x: (point.x - state.view.offsetX) / state.view.scale,
    y: (point.y - state.view.offsetY) / state.view.scale,
  };
}

function clampImagePoint(point) {
  return {
    x: Math.max(0, Math.min(state.imageWidth || state.image.width, point.x)),
    y: Math.max(0, Math.min(state.imageHeight || state.image.height, point.y)),
  };
}

function normalizedRect(start, end) {
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  return {
    x,
    y,
    w: Math.abs(end.x - start.x),
    h: Math.abs(end.y - start.y),
  };
}

function hitTest(point) {
  for (let i = state.detections.length - 1; i >= 0; i -= 1) {
    const detection = state.detections[i];
    const [x, y, w, h] = detection.bbox;
    if (point.x >= x && point.x <= x + w && point.y >= y && point.y <= y + h) {
      return detection.id;
    }
  }
  return -1;
}

async function addManualDetection(bbox, className) {
  if (!state.currentJobId) {
    setStatus("请先完成一次检测");
    return;
  }
  try {
    state.detections = await apiFetch(`/api/jobs/${state.currentJobId}/detections`, {
      method: "POST",
      body: JSON.stringify({ bbox, class_name: className }),
    });
    els.exportBtn.disabled = state.detections.length === 0;
    renderTable();
    drawCanvas();
    setStatus(`已添加手动目标: ${className}`);
  } catch (error) {
    setStatus(`手动添加失败: ${error.message}`);
  }
}

async function deleteDetection(index) {
  if (!state.currentJobId) return;
  try {
    state.detections = await apiFetch(`/api/jobs/${state.currentJobId}/detections/${index}`, {
      method: "DELETE",
    });
    state.selectedDetection = -1;
    els.exportBtn.disabled = state.detections.length === 0;
    renderTable();
    hideSegmentPanel();
    drawCanvas();
    setStatus("目标已删除");
  } catch (error) {
    setStatus(`删除失败: ${error.message}`);
  }
}

async function exportResults() {
  if (!state.currentJobId) return;
  try {
    const exported = await apiFetch(`/api/jobs/${state.currentJobId}/export`, {
      method: "POST",
      body: JSON.stringify({}),
    });
    const imageLink = document.createElement("a");
    imageLink.href = exported.image_url;
    imageLink.target = "_blank";
    imageLink.rel = "noreferrer";
    imageLink.textContent = "检测图像";

    const csvLink = document.createElement("a");
    csvLink.href = exported.csv_url;
    csvLink.target = "_blank";
    csvLink.rel = "noreferrer";
    csvLink.textContent = "结果CSV";

    els.exportLinks.innerHTML = "";
    els.exportLinks.appendChild(imageLink);
    els.exportLinks.appendChild(csvLink);
    setStatus(`结果已导出到 ${exported.output_dir}`);
  } catch (error) {
    setStatus(`导出失败: ${error.message}`);
  }
}

// ---------------------------------------------------------------------------
// File upload & delete
// ---------------------------------------------------------------------------

const CATEGORY_SELECTS = {
  detectors: "detectorSelect",
  classifiers: "classifierSelect",
  images: "imageSelect",
};

const FILE_INPUT_IDS = {
  detectors: "detectorFileInput",
  classifiers: "classifierFileInput",
  images: "imageFileInput",
};

function bindFileActions() {
  for (const [category, inputId] of Object.entries(FILE_INPUT_IDS)) {
    const fileInput = document.getElementById(inputId);
    if (!fileInput) continue;

    fileInput.addEventListener("change", () => uploadFile(category, fileInput));
  }

  for (const btn of document.querySelectorAll(".mini-upload")) {
    const category = btn.dataset.category;
    btn.addEventListener("click", () => {
      const fileInput = document.getElementById(FILE_INPUT_IDS[category]);
      if (fileInput) fileInput.click();
    });
  }

  for (const btn of document.querySelectorAll(".mini-delete")) {
    const category = btn.dataset.category;
    btn.addEventListener("click", () => deleteSelectedFile(category));
  }
}

async function uploadFile(category, fileInput) {
  const file = fileInput.files[0];
  if (!file) return;

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`/api/files/upload/${category}`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.detail || `${response.status} ${response.statusText}`);
    }

    const result = await response.json();
    setStatus(`已上传: ${result.filename}`);
    fileInput.value = "";
    await refreshFiles();

    // Auto-select the newly uploaded file
    const selectId = CATEGORY_SELECTS[category];
    if (selectId && els[selectId]) {
      els[selectId].value = result.filename;
      if (category === "images") {
        loadSelectedImage();
      } else {
        updateRunButton();
      }
    }
  } catch (error) {
    setStatus(`上传失败: ${error.message}`);
    fileInput.value = "";
  }
}

function deleteSelectedFile(category) {
  const selectId = CATEGORY_SELECTS[category];
  const select = els[selectId];
  if (!select) return;

  const filename = select.value;
  if (!filename) {
    setStatus("请先选择要删除的文件");
    return;
  }

  // Don't delete the demo detector
  if (category === "detectors" && filename === "__demo__") {
    setStatus("不能删除内置演示检测器");
    return;
  }

  const displayName = select.options[select.selectedIndex]?.textContent || filename;
  if (!window.confirm(`确定要删除 "${displayName}" 吗？此操作不可撤销。`)) {
    return;
  }

  apiFetch(`/api/files/${category}/${encodeURIComponent(filename)}`, {
    method: "DELETE",
  })
    .then((result) => {
      setStatus(`已删除: ${result.filename}`);
      return refreshFiles();
    })
    .catch((error) => {
      setStatus(`删除失败: ${error.message}`);
    });
}
