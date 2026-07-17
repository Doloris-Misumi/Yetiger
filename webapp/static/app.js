const state = {
  result: null,
  selectedFile: null,
  waveform: [],
  animationId: null,
  videoExporting: false,
  songVideoUrl: "",
  songVideoObjectUrl: "",
  songVideoName: "",
  songVideoFile: null,
  callAudioKey: "",
  actionLibrary: [],
  segmentDirty: false,
  timelineDirty: false,
  notesDirty: false,
};

const els = {
  form: document.getElementById("uploadForm"),
  audioInput: document.getElementById("audioInput"),
  titleInput: document.getElementById("titleInput"),
  fileName: document.getElementById("fileName"),
  dropZone: document.getElementById("dropZone"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  statusLine: document.getElementById("statusLine"),
  exampleSelect: document.getElementById("exampleSelect"),
  loadExampleBtn: document.getElementById("loadExampleBtn"),
  durationStat: document.getElementById("durationStat"),
  tempoStat: document.getElementById("tempoStat"),
  barsStat: document.getElementById("barsStat"),
  actionsStat: document.getElementById("actionsStat"),
  exportJsonBtn: document.getElementById("exportJsonBtn"),
  exportMdBtn: document.getElementById("exportMdBtn"),
  exportVideoBtn: document.getElementById("exportVideoBtn"),
  exportHint: document.getElementById("exportHint"),
  songVideoInput: document.getElementById("songVideoInput"),
  clearSongVideoBtn: document.getElementById("clearSongVideoBtn"),
  songVideoName: document.getElementById("songVideoName"),
  canvas: document.getElementById("videoCanvas"),
  audio: document.getElementById("audioPlayer"),
  timelineBody: document.getElementById("timelineBody"),
  currentAction: document.getElementById("currentAction"),
  saveBtn: document.getElementById("saveBtn"),
  saveNotesBtn: document.getElementById("saveNotesBtn"),
  rightPanel: document.getElementById("rightPanel"),
  structureEditor: document.getElementById("structureEditor"),
  segmentList: document.getElementById("segmentList"),
  addSegmentBtn: document.getElementById("addSegmentBtn"),
  editBadge: document.getElementById("editBadge"),
  timelineFilters: document.getElementById("timelineFilters"),
  geiVideoOverlay: document.getElementById("geiVideoOverlay"),
  geiVideoPlayer: document.getElementById("geiVideoPlayer"),
  songVideoPlayer: document.getElementById("songVideoPlayer"),
  callAudioPlayer: document.getElementById("callAudioPlayer"),
  notesList: document.getElementById("notesList"),
  addNoteBtn: document.getElementById("addNoteBtn"),
  notesBadge: document.getElementById("notesBadge"),
};

const GEI_VIDEO_MAP = {
  long_zhi_mao: "/api/gei-video/long_zhi_mao",
  lei_she: "/api/gei-video/lei_she",
};

const CALL_AUDIO_MAP = {
  standard_mix: "/api/call-audio/standard_mix",
  standard_mix_long: "/api/call-audio/standard_mix_long",
  standard_mix_first_half: "/api/call-audio/standard_mix_first_half",
  standard_mix_second_half: "/api/call-audio/standard_mix_second_half",
  mix_leadin_aaa_ikuzo: "/api/call-audio/mix_leadin_aaa_ikuzo",
  japanese_mix: "/api/call-audio/japanese_mix",
  japanese_mix_long: "/api/call-audio/japanese_mix_long",
  japanese_mix_second_half: "/api/call-audio/japanese_mix_second_half",
  ainu_mix: "/api/call-audio/ainu_mix",
  ainu_second_half_mix: "/api/call-audio/ainu_second_half_mix",
  ainu_kahen_mix: "/api/call-audio/ainu_kahen_mix",
  myhontousuke: "/api/call-audio/myhontousuke",
  myohon_activation: "/api/call-audio/myohon_activation",
  kaho_sanren_mix: "/api/call-audio/kaho_sanren_mix",
  ietora_konzetsu_mix: "/api/call-audio/ietora_konzetsu_mix",
  ietora: "/api/call-audio/ietora",
  ietora_long: "/api/call-audio/ietora_long",
  tiger_fire_activation: "/api/call-audio/tiger_fire_activation",
  bismarck_mix: "/api/call-audio/bismarck_mix",
  bismarck_mix_first_half: "/api/call-audio/bismarck_mix_first_half",
  sekai_konton_mix: "/api/call-audio/sekai_konton_mix",
  sekai_konton_mix_first_half: "/api/call-audio/sekai_konton_mix_first_half",
  bandor_mix: "/api/call-audio/bandor_mix",
  popipa_mix: "/api/call-audio/popipa_mix",
  lin_xiu_mix: "/api/call-audio/lin_xiu_mix",
  bariyado_mix: "/api/call-audio/bariyado_mix",
  pan_mix: "/api/call-audio/pan_mix",
  aiai_mix: "/api/call-audio/aiai_mix",
  imi_fumei_ai_mix: "/api/call-audio/imi_fumei_ai_mix",
};
const missingCallAudioIds = new Set();
const CALL_AUDIO_VERSION = "20260717-duration-fit";

function resolveApiBase() {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("api");
  if (fromQuery) {
    const normalized = fromQuery.replace(/\/$/, "");
    localStorage.setItem("yestiger_api_base", normalized);
    return normalized;
  }
  return (window.YESTIGER_API_BASE || localStorage.getItem("yestiger_api_base") || "").replace(/\/$/, "");
}

const API_BASE = resolveApiBase();

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

async function fetchJson(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 90000); // 90s for cold start
  try {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...options,
    });
    clearTimeout(timeout);
    const data = await response.json();
    if (!response.ok || data.error) {
      throw new Error(data.message || data.error || `Request failed: ${response.status}`);
    }
    return data;
  } catch (e) {
    clearTimeout(timeout);
    if (e.name === "AbortError") throw new Error("服务器响应超时（可能正在唤醒，请刷新后重试）");
    throw e;
  }
}

async function wakeUpBackend() {
  if (!API_BASE) return true;
  setStatus("正在连接服务器...");
  for (let i = 0; i < 3; i++) {
    try {
      await fetchJson(apiUrl("/api/actions"));
      setStatus("就绪");
      return true;
    } catch (e) {
      if (i < 2) {
        setStatus(`正在唤醒服务器... (${i + 1}/3)`);
        await new Promise(r => setTimeout(r, 5000));
      }
    }
  }
  setStatus("无法连接服务器，请稍后刷新页面重试");
  return false;
}

const roleColors = {
  keepspace: "#6b7280",
  rhythmcall: "#1d8f74",
  mix: "#c65347",
  underground_gei: "#7a4fa3",
};

const musicColors = {
  intro: "#2f6fb2",
  verse: "#1d8f74",
  pre_chorus: "#b7791f",
  pre_chorus_build: "#d97706",
  chorus: "#c65347",
  post_chorus: "#9f5f2a",
  bridge: "#7a4fa3",
  instrumental: "#0f766e",
  instrumental_break: "#0f766e",
  interlude: "#0f766e",
  solo: "#8b5cf6",
  outro: "#475467",
  end: "#334155",
  unknown: "#64748b",
};

const musicLabels = {
  intro: "前奏",
  verse: "主歌",
  pre_chorus: "预副歌",
  pre_chorus_build: "预副歌推进",
  chorus: "副歌",
  post_chorus: "后副歌",
  bridge: "桥段",
  instrumental: "间奏/纯音乐",
  instrumental_break: "间奏 Break",
  interlude: "间奏",
  solo: "Solo",
  outro: "尾奏",
  end: "结束",
  unknown: "未知段落",
};

const roleLabels = {
  keepspace: "留白",
  rhythmcall: "节奏 Call",
  mix: "MIX",
  underground_gei: "地下艺",
};

const riskLabels = {
  low: "低风险",
  medium: "中风险",
  high: "高风险",
};

function setStatus(text) {
  els.statusLine.textContent = text;
}

function fmtTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const secs = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${secs.toFixed(2).padStart(5, "0")}`;
}

function parseTimeInput(value) {
  const text = String(value || "").trim().replace("：", ":");
  const match = text.match(/^(\d+):(\d+(?:\.\d+)?)$/);
  if (match) return parseInt(match[1], 10) * 60 + parseFloat(match[2]);
  const seconds = Number(text);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : null;
}

function musicLabelText(label) {
  const key = String(label || "unknown").trim();
  if (!key) return musicLabels.unknown;
  return musicLabels[key] || key.replaceAll("_", " ");
}

function downloadText(filename, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function hasAudioSource() {
  return Boolean(els.audio.currentSrc || els.audio.src);
}

function updateExportAvailability(message) {
  const canExportMp4 = Boolean(state.result && hasAudioSource());
  els.exportVideoBtn.disabled = state.videoExporting || !canExportMp4;
  els.exportVideoBtn.textContent = state.videoExporting ? "正在生成 MP4..." : "下载教学视频 (.mp4)";
  if (!els.exportHint) return;
  if (state.videoExporting) {
    els.exportHint.textContent = "正在后端渲染画面并合成音频";
  } else if (message) {
    els.exportHint.textContent = message;
  } else if (!state.result) {
    els.exportHint.textContent = "请先加载分析";
  } else if (!hasAudioSource()) {
    els.exportHint.textContent = "请先加载音频";
  } else {
    els.exportHint.textContent = state.songVideoUrl
      ? "MP4 会包含右上角歌曲视频、原曲音频和 call 声音。"
      : "MP4 会在后端离线渲染，并合成原曲音频和 call 声音。";
  }
}

function updateSongVideoUi() {
  if (!els.songVideoName || !els.clearSongVideoBtn) return;
  els.songVideoName.textContent = state.songVideoName || "未加载歌曲视频";
  els.clearSongVideoBtn.disabled = !state.songVideoUrl;
}

function clearSongVideo(options = {}) {
  if (state.songVideoObjectUrl) {
    URL.revokeObjectURL(state.songVideoObjectUrl);
  }
  state.songVideoUrl = "";
  state.songVideoObjectUrl = "";
  state.songVideoName = "";
  state.songVideoFile = null;
  if (els.songVideoInput) els.songVideoInput.value = "";
  if (els.songVideoPlayer) {
    els.songVideoPlayer.pause();
    els.songVideoPlayer.removeAttribute("src");
    els.songVideoPlayer.style.display = "none";
    els.songVideoPlayer.load();
  }
  updateSongVideoUi();
  updateExportAvailability();
  if (!options.silent) setStatus("歌曲视频已移除");
}

function setSongVideoFile(file) {
  if (!file) return;
  if (file.type && !file.type.startsWith("video/")) {
    setStatus("请选择视频文件");
    return;
  }
  clearSongVideo({ silent: true });
  const url = URL.createObjectURL(file);
  state.songVideoUrl = url;
  state.songVideoObjectUrl = url;
  state.songVideoName = file.name;
  state.songVideoFile = file;
  els.songVideoPlayer.src = url;
  els.songVideoPlayer.load();
  updateSongVideoUi();
  updateExportAvailability("歌曲视频已加载，会跟随音频播放与拖动");
  setStatus("歌曲视频已加载");
  syncSongVideoToAudio(true);
}

function applyVideoOverlayBounds() {
  const overlay = els.geiVideoOverlay;
  const canvasRect = els.canvas.getBoundingClientRect();
  const parentRect = els.canvas.parentElement.getBoundingClientRect();
  const gap = 6;
  const leftW = 386;
  const topH = 430;
  const scaleX = canvasRect.width / 1280;
  const scaleY = canvasRect.height / 720;
  const mediaX = canvasRect.left - parentRect.left + (leftW + gap) * scaleX;
  const mediaY = canvasRect.top - parentRect.top;
  const mediaW = (1280 - leftW - gap) * scaleX;
  const mediaH = topH * scaleY;

  overlay.style.left = `${mediaX}px`;
  overlay.style.top = `${mediaY}px`;
  overlay.style.width = `${mediaW}px`;
  overlay.style.height = `${mediaH}px`;
}

function pauseOverlayVideos() {
  els.geiVideoPlayer.pause();
  stopCallAudio();
  if (els.songVideoPlayer) els.songVideoPlayer.pause();
  if (!state.songVideoUrl) els.geiVideoOverlay.hidden = true;
}

function syncSongVideoToAudio(force = false) {
  if (!state.songVideoUrl || !els.songVideoPlayer) return false;
  const video = els.songVideoPlayer;
  const overlay = els.geiVideoOverlay;

  applyVideoOverlayBounds();
  els.geiVideoPlayer.pause();
  els.geiVideoPlayer.style.display = "none";
  video.style.display = "block";
  overlay.hidden = false;

  if (!video.src) {
    video.src = state.songVideoUrl;
    video.load();
  }

  const targetTime = Math.max(0, Number(els.audio.currentTime || 0));
  if (video.readyState >= 1 && Number.isFinite(video.duration) && video.duration > 0) {
    const boundedTime = Math.min(targetTime, Math.max(0, video.duration - 0.05));
    if (force || Math.abs(video.currentTime - boundedTime) > 0.35) {
      try {
        video.currentTime = boundedTime;
      } catch (_error) {
        // Some browsers reject early seeks before metadata is ready.
      }
    }
  }

  if (Math.abs(video.playbackRate - els.audio.playbackRate) > 0.01) {
    video.playbackRate = els.audio.playbackRate;
  }

  if (els.audio.paused || els.audio.ended || !hasAudioSource()) {
    video.pause();
  } else if (video.paused) {
    video.play().catch(() => {});
  }
  return true;
}

function updateOverlayVideo(current, time) {
  if (syncSongVideoToAudio()) return;
  updateGeiVideo(current, time);
}

function filenameFromDisposition(value, fallback) {
  const header = String(value || "");
  const utf8Match = header.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match) return decodeURIComponent(utf8Match[1]);
  const plainMatch = header.match(/filename="?([^";]+)"?/i);
  return plainMatch ? plainMatch[1] : fallback;
}

function editableResult() {
  if (!state.result) return null;
  return JSON.parse(JSON.stringify(state.result));
}

function markdownFromTimeline(result) {
  const title = result?.song?.title || "YesTiger Callbook";
  const lines = [
    `# ${title}`,
    "",
    "| Time | Music | Struct | Role | Action | Bars | Risk | Text |",
    "|---:|---|---|---|---|---:|---|---|",
  ];
  for (const action of result.timeline || []) {
    const music = musicLabelText(action.music_label);
    const struct = musicLabelText(action.struct_label);
    lines.push(
      `| ${fmtTime(action.start)}-${fmtTime(action.end)} | ${music} | ${struct} | ${action.role || "-"} | ${action.display_name || "-"} | ${action.bar_count ?? "-"} | ${action.risk || "-"} | ${action.typical_text || "-"} |`
    );
  }
  const notes = Array.isArray(result.notes) ? result.notes.filter((note) => String(note.text || "").trim()) : [];
  if (notes.length) {
    lines.push("", "## 备注", "");
    for (const note of notes) {
      lines.push(`- ${fmtTime(note.start)}-${fmtTime(note.end)} ${String(note.text || "").trim()}`);
    }
  }
  return `${lines.join("\n")}\n`;
}

function actionAtFromTimeline(time, timeline) {
  return timeline.find((item) => time >= Number(item.start) && time < Number(item.end)) || null;
}

function actionAt(time) {
  return actionAtFromTimeline(time, state.result?.timeline || []);
}

function nextAction(time) {
  const timeline = state.result?.timeline || [];
  return timeline.find((item) => Number(item.start) > time) || null;
}

function musicSegmentAt(time) {
  const segments = state.result?.music_segments || state.result?.segments || [];
  return segments.find((item) => time >= Number(item.start) && time < Number(item.end)) || null;
}

function tutorialBars(action) {
  const bars = action?.tutorial_text?.bars;
  return Array.isArray(bars) ? bars.filter((item) => String(item || "").trim()) : [];
}

function currentTutorialCue(action, time) {
  const bars = tutorialBars(action);
  if (!action || !bars.length) return null;
  const start = Number(action.start) || 0;
  const end = Math.max(start + 0.001, Number(action.end) || start + 0.001);
  const progress = clamp((time - start) / (end - start), 0, 0.999999);
  const index = clamp(Math.floor(progress * bars.length), 0, bars.length - 1);
  return {
    index,
    total: bars.length,
    text: bars[index],
    source: action.tutorial_text?.source,
  };
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

function withAlpha(hex, alpha) {
  const clean = String(hex || "#000000").replace("#", "");
  if (clean.length !== 6) return `rgba(255,255,255,${alpha})`;
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function wrapCanvasText(ctx, text, maxWidth) {
  const source = String(text || "").replace(/\s+/g, " ").trim();
  if (!source) return [];
  const tokens = source.match(/[\u3040-\u30ff\u3400-\u9fff\uff00-\uffef]|[^\s\u3040-\u30ff\u3400-\u9fff\uff00-\uffef]+/g) || [];
  const lines = [];
  let line = "";
  tokens.forEach((token) => {
    const glue = line && !/^[\u3040-\u30ff\u3400-\u9fff\uff00-\uffef]$/.test(token) ? " " : "";
    const next = `${line}${glue}${token}`;
    if (ctx.measureText(next).width <= maxWidth || !line) {
      line = next;
    } else {
      lines.push(line);
      line = token;
    }
  });
  if (line) lines.push(line);
  return lines;
}

function drawWrappedText(ctx, text, x, y, maxWidth, lineHeight, maxLines) {
  const lines = wrapCanvasText(ctx, text, maxWidth).slice(0, maxLines);
  lines.forEach((line, index) => {
    const suffix = index === maxLines - 1 && wrapCanvasText(ctx, text, maxWidth).length > maxLines ? "..." : "";
    ctx.fillText(`${line}${suffix}`, x, y + index * lineHeight);
  });
  return lines.length;
}

function fitCanvasFont(ctx, text, weight, size, family, maxWidth, minSize = 22) {
  let current = size;
  do {
    ctx.font = `${weight} ${current}px ${family}`;
    if (ctx.measureText(String(text || "")).width <= maxWidth) return current;
    current -= 2;
  } while (current >= minSize);
  ctx.font = `${weight} ${minSize}px ${family}`;
  return minSize;
}

function renderStats() {
  const song = state.result?.song || {};
  els.durationStat.textContent = song.duration ? fmtTime(song.duration) : "-";
  els.tempoStat.textContent = song.tempo ? `${Math.round(song.tempo)} BPM` : "-";
  els.barsStat.textContent = song.bar_count ?? "-";
  els.actionsStat.textContent = (state.result?.timeline || []).length || "-";
}

function renderProcess() {
  // Keep pipeline fields in saved analysis results, but hide debug-oriented
  // process details from the product UI.
}

function renderTimeline() {
  const timeline = state.result?.timeline || [];
  els.timelineBody.innerHTML = "";
  const activeRole = els.timelineFilters.dataset.activeRole || "all";
  timeline.forEach((action, index) => {
    if (activeRole !== "all" && action.role !== activeRole) return;
    const tr = document.createElement("tr");
    tr.dataset.index = String(index);
    const riskClass = action.risk === "high" ? "risk-high" : action.risk === "medium" ? "risk-medium" : "";
    const roleLabel = roleLabels[action.role] || action.role || "-";
    const hasGeiVideo = action.action_id && GEI_VIDEO_MAP[action.action_id];
    tr.innerHTML = `
      <td class="row-num">${index + 1}</td>
      <td class="time-cell">
        <input class="time-inline" value="${fmtTime(action.start)}" data-index="${index}" data-field="start" size="7" />
        <span class="time-dash">-</span>
        <input class="time-inline" value="${fmtTime(action.end)}" data-index="${index}" data-field="end" size="7" />
      </td>
      <td><span class="music-pill music-${action.music_label || "unknown"}">${escapeHtml(musicLabelText(action.music_label))}</span></td>
      <td><span class="role-pill role-${action.role || "keepspace"}">${escapeHtml(roleLabel)}</span></td>
      <td class="action-cell">
        <input class="action-search" value="${escapeAttr(action.display_name || "")}" data-index="${index}" data-field="display_name" placeholder="Search action..." autocomplete="off" />
        <ul class="action-dropdown" hidden></ul>
      </td>
      <td>${action.bar_count ?? "-"}</td>
      <td class="${riskClass}">${action.risk || "low"}</td>
      <td class="text-cell">${escapeHtml(action.typical_text || "")}</td>
      <td class="row-actions">
        <button class="row-btn row-btn-add" data-index="${index}" title="Insert action after this">+</button>
        <button class="row-btn row-btn-del" data-index="${index}" title="Remove this action">&times;</button>
      </td>
    `;
    if (hasGeiVideo) tr.classList.add("has-gei-video");
    els.timelineBody.appendChild(tr);
  });
  bindActionSearchInputs();
  bindRowButtons();
}

function bindRowButtons() {
  els.timelineBody.querySelectorAll(".row-btn-add").forEach((btn) => {
    btn.removeEventListener("click", handleInsertAction);
    btn.addEventListener("click", handleInsertAction);
  });
  els.timelineBody.querySelectorAll(".row-btn-del").forEach((btn) => {
    btn.removeEventListener("click", handleDeleteAction);
    btn.addEventListener("click", handleDeleteAction);
  });
}

function handleInsertAction(e) {
  const index = Number(e.target.dataset.index);
  const timeline = state.result.timeline;
  const current = timeline[index];
  const next = timeline[index + 1];
  const midStart = Number(current.end) || (Number(current.start) + 4);
  const maxEnd = next ? Number(next.start) : (midStart + 16);
  const newEnd = Math.min(midStart + 8, maxEnd);
  timeline.splice(index + 1, 0, {
    start: Math.round(midStart * 100) / 100,
    end: Math.round(newEnd * 100) / 100,
    time: `${fmtTime(midStart)}-${fmtTime(newEnd)}`,
    action_id: "",
    display_name: "留白",
    role: "keepspace",
    music_label: current.music_label,
    struct_label: current.struct_label,
    risk: "low",
    bar_count: null,
    typical_text: "",
    tutorial_text: null,
    confidence: null,
    mode: "human_curated",
    notes: "",
  });
  recalcAllBarCounts();
  state.timelineDirty = true;
  renderTimeline();
}

function handleDeleteAction(e) {
  const index = Number(e.target.dataset.index);
  const timeline = state.result.timeline;
  if (timeline.length <= 1) return;
  timeline.splice(index, 1);
  recalcAllBarCounts();
  state.timelineDirty = true;
  renderTimeline();
}

function recalcBarCount(action) {
  const downbeats = getDownbeats();
  const start = Number(action.start) || 0;
  const end = Number(action.end) || start;
  if (end <= start) {
    action.bar_count = null;
    return;
  }
  const barSeconds = inferredBarSeconds();
  if (downbeats.length >= 2) {
    const startIndex = nearestDownbeatIndex(start, downbeats);
    const endIndex = nearestDownbeatIndex(end, downbeats);
    const startDist = Math.abs(downbeats[startIndex] - start);
    const endDist = Math.abs(downbeats[endIndex] - end);
    const tolerance = Math.max(0.18, (Number.isFinite(barSeconds) ? barSeconds : 2.4) * 0.38);
    if (endIndex > startIndex && startDist <= tolerance && endDist <= tolerance) {
      action.bar_count = endIndex - startIndex;
      return;
    }
  }
  if (Number.isFinite(barSeconds) && barSeconds > 0) {
    action.bar_count = Math.max(1, Math.round((end - start) / barSeconds));
    return;
  }
  const intervals = [];
  for (let i = 0; i < downbeats.length - 1; i++) {
    const length = downbeats[i + 1] - downbeats[i];
    if (length > 0.2 && length < 20) intervals.push(length);
  }
  intervals.sort((a, b) => a - b);
  const median = intervals.length ? intervals[Math.floor(intervals.length / 2)] : 0;
  action.bar_count = median > 0 ? Math.max(1, Math.round((end - start) / median)) : null;
}

function recalcAllBarCounts() {
  for (const item of state.result.timeline || []) {
    recalcBarCount(item);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("'", "&#39;");
}

function updateActiveRow() {
  const time = els.audio.currentTime || 0;
  const active = actionAt(time);
  const rows = els.timelineBody.querySelectorAll("tr");
  rows.forEach((row) => row.classList.remove("is-active"));
  if (active) {
    const index = (state.result.timeline || []).indexOf(active);
    const row = els.timelineBody.querySelector(`tr[data-index="${index}"]`);
    if (row) row.classList.add("is-active");
    els.currentAction.textContent = `${musicLabelText(active.music_label)} | ${active.role || "-"} | ${active.display_name} | ${fmtTime(active.start)}-${fmtTime(active.end)}`;
  } else {
    els.currentAction.textContent = state.result ? "留白" : "未加载动作";
  }
}

async function buildWaveformFromUrl(url) {
  try {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    await buildWaveform(buffer);
  } catch (error) {
    state.waveform = [];
  }
}

async function buildWaveform(buffer) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    state.waveform = [];
    return;
  }
  const context = new AudioContextClass();
  const audioBuffer = await context.decodeAudioData(buffer.slice(0));
  const data = audioBuffer.getChannelData(0);
  const buckets = 240;
  const step = Math.max(1, Math.floor(data.length / buckets));
  const values = [];
  for (let i = 0; i < buckets; i += 1) {
    let sum = 0;
    const start = i * step;
    const end = Math.min(data.length, start + step);
    for (let j = start; j < end; j += 1) sum += Math.abs(data[j]);
    values.push(sum / Math.max(1, end - start));
  }
  const max = Math.max(...values, 0.0001);
  state.waveform = values.map((value) => value / max);
  if (context.close) context.close();
}

function drawCanvas() {
  const canvas = els.canvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const result = state.result;
  const time = els.audio.currentTime || 0;
  const duration = Number(result?.song?.duration || els.audio.duration || 1);
  const current = actionAt(time);
  const upcoming = nextAction(time);
  const currentMusic = musicSegmentAt(time);

  updateOverlayVideo(current, time);
  updateCallAudio(current, time);

  drawTeachingCanvas(ctx, width, height, result, current, upcoming, currentMusic, duration, time);

  updateActiveRow();
  state.animationId = requestAnimationFrame(drawCanvas);
}

function updateGeiVideo(current, time) {
  const actionId = current?.action_id;
  const videoSrc = GEI_VIDEO_MAP[actionId];
  const video = els.geiVideoPlayer;
  const overlay = els.geiVideoOverlay;

  if (!videoSrc || !current) {
    overlay.hidden = true;
    video.pause();
    video.style.display = "none";
    video.removeAttribute("src");
    return;
  }

  applyVideoOverlayBounds();
  if (els.songVideoPlayer) {
    els.songVideoPlayer.pause();
    els.songVideoPlayer.style.display = "none";
  }
  video.style.display = "block";

  const currentSrc = video.getAttribute("src") || "";
  if (!currentSrc.includes(videoSrc)) {
    video.src = videoSrc;
    video.load();
  }

  const actionStart = Number(current.start) || 0;
  const actionEnd = Number(current.end) || actionStart + 1;
  const actionDuration = Math.max(0.1, actionEnd - actionStart);
  const localTime = Math.max(0, Math.min(actionDuration, time - actionStart));

  if (video.readyState >= 2 && video.duration > 0) {
    const targetRate = Math.max(0.75, Math.min(4.0, video.duration / actionDuration));
    if (Math.abs(video.playbackRate - targetRate) > 0.01) {
      video.playbackRate = targetRate;
    }
    const expectedTime = (localTime / actionDuration) * video.duration;
    if (Math.abs(video.currentTime - expectedTime) > 0.25) {
      video.currentTime = expectedTime;
    }
  }

  if (video.paused) {
    video.play().catch(() => {});
  }

  overlay.hidden = false;
}

function stopCallAudio({ clearSource = false, resetTime = true } = {}) {
  const audio = els.callAudioPlayer;
  const hadSource = Boolean(audio.getAttribute("src"));
  if (!hadSource && !state.callAudioKey && audio.paused) return;
  audio.pause();
  if (resetTime) {
    try {
      audio.currentTime = 0;
    } catch (_error) {
      // The media element may not have metadata yet.
    }
  }
  if (clearSource && hadSource) {
    audio.removeAttribute("src");
    audio.load();
  }
  state.callAudioKey = "";
}

function updateCallAudio(current, time) {
  const actionId = current?.action_id;
  const audioSrc = callAudioUrlForAction(current);
  const audio = els.callAudioPlayer;

  if (!audioSrc || !current) {
    stopCallAudio({ clearSource: true });
    return;
  }

  const actionStart = Number(current.start) || 0;
  const actionEnd = Number(current.end) || actionStart + 1;
  const actionDuration = Math.max(0.1, actionEnd - actionStart);
  if (time < actionStart || time >= actionEnd - 0.025) {
    stopCallAudio();
    return;
  }

  const resolvedAudioSrc = apiUrl(audioSrc);
  const currentSrc = audio.getAttribute("src") || "";
  const actionKey = `${actionId}:${actionStart.toFixed(3)}:${actionEnd.toFixed(3)}`;
  if (!currentSrc.includes(resolvedAudioSrc) || state.callAudioKey !== actionKey) {
    audio.pause();
    audio.src = resolvedAudioSrc;
    state.callAudioKey = actionKey;
    audio.onerror = () => {
      if (actionId) missingCallAudioIds.add(actionId);
      stopCallAudio({ clearSource: true });
    };
    audio.load();
    try {
      audio.currentTime = 0;
    } catch (_error) {
      // Metadata may not be ready yet.
    }
  }

  const localTime = Math.max(0, Math.min(actionDuration, time - actionStart));

  if (audio.readyState >= 2 && audio.duration > 0) {
    const targetRate = Math.max(0.5, Math.min(8.0, audio.duration / actionDuration));
    if (Math.abs(audio.playbackRate - targetRate) > 0.01) {
      audio.playbackRate = targetRate;
    }
    const expectedTime = (localTime / actionDuration) * audio.duration;
    if (expectedTime >= audio.duration - 0.025) {
      stopCallAudio();
      return;
    }
    if (Math.abs(audio.currentTime - expectedTime) > 0.12) {
      audio.currentTime = expectedTime;
    }
  }

  if (audio.paused) {
    audio.play().catch(() => {});
  }
}

function callAudioUrlForAction(current) {
  const actionId = current?.action_id;
  if (!actionId || missingCallAudioIds.has(actionId)) return null;
  const baseUrl = CALL_AUDIO_MAP[actionId] || (current?.role === "mix" ? `/api/call-audio/${encodeURIComponent(actionId)}` : null);
  if (!baseUrl) return null;
  const separator = baseUrl.includes("?") ? "&" : "?";
  return `${baseUrl}${separator}v=${CALL_AUDIO_VERSION}`;
}

function drawTeachingCanvas(ctx, width, height, result, current, upcoming, currentMusic, duration, time) {
  const gap = 6;
  const leftW = 386;
  const topH = 430;
  const rightW = width - leftW - gap;
  const bottomH = height - topH - gap;
  const panels = {
    action: { x: 0, y: 0, w: leftW, h: height },
    media: { x: leftW + gap, y: 0, w: rightW, h: topH },
    method: { x: leftW + gap, y: topH + gap, w: rightW, h: bottomH },
  };
  const role = current?.role || "keepspace";
  const roleColor = roleColors[role] || roleColors.keepspace;

  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, width, height);
  Object.values(panels).forEach((panel) => {
    ctx.fillStyle = "#050505";
    ctx.fillRect(panel.x, panel.y, panel.w, panel.h);
  });

  drawActionPanel(ctx, panels.action, current, roleColor, result, currentMusic, duration, time, upcoming);
  drawMediaPanel(ctx, panels.media, result, current, currentMusic, roleColor);
  drawMethodPanel(ctx, panels.method, current, time, duration, roleColor);

  // Update HTML song info bar
  updateSongInfoBar(result, current, upcoming, currentMusic, duration, time);
}

function drawPanelLabel(ctx, panel, label, color = "#a8b3c2") {
  ctx.fillStyle = color;
  ctx.font = "700 18px Segoe UI, sans-serif";
  ctx.fillText(label, panel.x + 26, panel.y + 40);
}

function drawNotePanel(ctx, panel, result, current, upcoming, currentMusic, duration, time, roleColor) {
  const x = panel.x + 30;
  const maxW = panel.w - 60;
  drawPanelLabel(ctx, panel, "备注", "#e5e7eb");

  const title = result?.song?.title || "YesTiger";
  fitCanvasFont(ctx, title, "800", 40, "Segoe UI, sans-serif", maxW, 24);
  ctx.fillStyle = "#f8fafc";
  drawWrappedText(ctx, title, x, panel.y + 92, maxW, 44, 2);

  const section = musicLabelText(currentMusic?.music_label || current?.music_label);
  const actionWindow = current
    ? `${fmtTime(current.start)}-${fmtTime(current.end)}`
    : `Now ${fmtTime(time)}`;
  const infoRows = [
    `当前时间  ${fmtTime(time)} / ${fmtTime(duration)}`,
    `段落  ${section}`,
    `动作区间  ${actionWindow}`,
    `小节  ${current?.bar_count ?? "-"} bars`,
  ];

  ctx.font = "24px Segoe UI, sans-serif";
  ctx.fillStyle = "#d1d5db";
  infoRows.forEach((row, index) => {
    ctx.fillText(row, x, panel.y + 192 + index * 36);
  });

  if (upcoming) {
    ctx.fillStyle = "#9ca3af";
    ctx.font = "20px Segoe UI, sans-serif";
    drawWrappedText(
      ctx,
      `Next  ${fmtTime(upcoming.start)}  ${upcoming.display_name || "-"}`,
      x,
      panel.y + panel.h - 38,
      maxW,
      24,
      1
    );
  }
}

function drawMediaPanel(ctx, panel, result, current, currentMusic, roleColor) {
  const accent = musicColors[currentMusic?.music_label] || roleColor;
  const hasGeiVideo = current?.action_id && GEI_VIDEO_MAP[current.action_id];

  if (hasGeiVideo) {
    ctx.fillStyle = "#000000";
    ctx.fillRect(panel.x, panel.y, panel.w, panel.h);
    ctx.fillStyle = roleColor;
    ctx.fillRect(panel.x, panel.y + panel.h - 6, panel.w, 6);
    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "center";
    ctx.font = "700 32px Segoe UI, Microsoft YaHei, sans-serif";
    ctx.fillText(`${current.display_name} · 演示动作`, panel.x + panel.w / 2, panel.y + 50);
    ctx.font = "20px Segoe UI, sans-serif";
    ctx.fillText("视频同步播放中", panel.x + panel.w / 2, panel.y + panel.h - 36);
    ctx.textAlign = "left";
    return;
  }

  ctx.fillStyle = withAlpha(accent, 0.9);
  ctx.fillRect(panel.x, panel.y, panel.w, panel.h);

  ctx.fillStyle = "rgba(255, 255, 255, 0.16)";
  for (let col = 0; col < 16; col += 1) {
    const size = 14 + (col % 3) * 8;
    ctx.fillRect(panel.x + 40 + col * 58, panel.y + 56 + (col % 5) * 52, size, size);
  }

  ctx.fillStyle = "rgba(0, 0, 0, 0.18)";
  ctx.fillRect(panel.x, panel.y + panel.h - 108, panel.w, 108);

  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "center";
  fitCanvasFont(ctx, "MV / DEMO SLOT", "900", 66, "Segoe UI, sans-serif", panel.w - 100, 32);
  ctx.fillText("MV / DEMO SLOT", panel.x + panel.w / 2, panel.y + panel.h / 2 - 10);

  ctx.font = "24px Segoe UI, sans-serif";
  const caption = `${result?.song?.title || "YesTiger"} · ${musicLabelText(currentMusic?.music_label || current?.music_label)}`;
  ctx.fillText(caption, panel.x + panel.w / 2, panel.y + panel.h - 42);
  ctx.textAlign = "left";
}

function drawActionPanel(ctx, panel, current, roleColor, result, currentMusic, duration, time, upcoming) {
  const x = panel.x + 30;
  const maxW = panel.w - 60;

  // Top section: time-based notes only (no auto song info)
  const note = currentNote(time);
  let infoTopY = panel.y + 50;

  if (note) {
    ctx.fillStyle = "#f8fafc";
    fitCanvasFont(ctx, note.text, "800", 48, "Segoe UI, Microsoft YaHei, sans-serif", maxW, 30);
    const noteLines = wrapCanvasText(ctx, note.text, maxW).slice(0, 8);
    const noteLineH = Math.max(40, Number(ctx.font.match(/(\d+)px/)?.[1] || 40) + 10);
    ctx.fillStyle = "#e5e7eb";
    noteLines.forEach((line, i) => {
      ctx.fillText(line, x, panel.y + 42 + i * noteLineH);
    });
    infoTopY = panel.y + 42 + noteLines.length * noteLineH + 24;

    // Divider
    ctx.fillStyle = "#2a2a2a";
    ctx.fillRect(x, infoTopY, maxW, 1);
    infoTopY += 18;
  }

  // Divider
  ctx.fillStyle = "#2a2a2a";
  ctx.fillRect(x, panel.y + 360, maxW, 2);

  // Bottom section: action info
  const actionTop = panel.y + 390;
  drawPanelLabel(ctx, { x: panel.x, y: actionTop, w: panel.w, h: 0 }, "应援种类及名称", "#e5e7eb");

  ctx.fillStyle = roleColor;
  ctx.fillRect(x, actionTop + 46, maxW, 6);

  const roleText = roleLabels[current?.role] || roleLabels.keepspace;
  fitCanvasFont(ctx, roleText, "900", 44, "Segoe UI, Microsoft YaHei, sans-serif", maxW, 24);
  ctx.fillStyle = "#f8fafc";
  ctx.fillText(roleText, x, actionTop + 100);

  const actionName = current?.display_name || "留白";
  ctx.font = "800 36px Segoe UI, Microsoft YaHei, sans-serif";
  ctx.fillStyle = "#f8fafc";
  drawWrappedText(ctx, actionName, x, actionTop + 152, maxW, 42, 2);

  ctx.font = "20px Segoe UI, sans-serif";
  ctx.fillStyle = "#a8b3c2";
  const meta = current
    ? `${musicLabelText(current.music_label)} · ${current.bar_count ?? "-"} bars · ${riskLabels[current.risk] || current.risk || "low"}`
    : "未加载动作";
  drawWrappedText(ctx, meta, x, panel.y + panel.h - 50, maxW, 24, 1);

  // Upcoming
  if (upcoming) {
    ctx.fillStyle = "#9ca3af";
    ctx.font = "18px Segoe UI, sans-serif";
    drawWrappedText(
      ctx,
      `Next  ${fmtTime(upcoming.start)}  ${upcoming.display_name || "-"}`,
      x,
      panel.y + panel.h - 22,
      maxW,
      22,
      1
    );
  }
}

function updateSongInfoBar(result, current, upcoming, currentMusic, duration, time) {
  if (!els.songInfoBar) return;
  const section = musicLabelText(currentMusic?.music_label || current?.music_label);
  const upcomingText = upcoming
    ? `Next: ${fmtTime(upcoming.start)} ${upcoming.display_name || "-"}`
    : "";
  els.songInfoBar.innerHTML = [
    `<span>⏱ ${fmtTime(time)} / ${fmtTime(duration)}</span>`,
    `<span>🎵 ${section}</span>`,
    `<span>📏 ${current?.bar_count ?? "-"} bars</span>`,
    upcomingText ? `<span>⏭ ${upcomingText}</span>` : "",
  ].filter(Boolean).join(" &nbsp;|&nbsp; ");
}

function drawMethodPanel(ctx, panel, current, time, duration, roleColor) {
  const x = panel.x + 54;
  const maxW = panel.w - 108;
  drawPanelLabel(ctx, panel, "具体打法", "#e5e7eb");

  const cue = currentTutorialCue(current, time);
  const text = cue?.text || current?.typical_text || (current ? `${current.display_name || "Action"}：按当前段落节拍执行。` : "");
  if (cue) {
    ctx.fillStyle = "#a8b3c2";
    ctx.font = "700 22px Segoe UI, sans-serif";
    const cueLabel = `Bar cue ${cue.index + 1}/${cue.total}${cue.source ? ` · ${cue.source}` : ""}`;
    ctx.fillText(cueLabel, x, panel.y + 78);
  }

  const family = "Georgia, 'Times New Roman', Microsoft YaHei, serif";
  fitCanvasFont(ctx, text, "900", 44, family, maxW, 28);
  ctx.fillStyle = "#f8fafc";
  ctx.textAlign = "center";
  const lineHeight = Math.max(36, Number(ctx.font.match(/(\d+)px/)?.[1] || 36) + 12);
  const lines = wrapCanvasText(ctx, text, maxW).slice(0, 4);
  const startY = panel.y + 104 + Math.max(0, (panel.h - 172 - lines.length * lineHeight) / 2);
  lines.forEach((line, index) => {
    ctx.fillText(line, panel.x + panel.w / 2, startY + index * lineHeight);
  });
  ctx.textAlign = "left";
}

function drawMusicBands(ctx, width, height, duration, time) {
  const segments = state.result?.music_segments || state.result?.segments || [];
  const left = 52;
  const top = height - 226;
  const w = width - 104;
  const h = 24;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(left, top, w, h);
  segments.forEach((segment) => {
    const label = segment.music_label || "unknown";
    const x = left + (Number(segment.start) / duration) * w;
    const endX = left + (Number(segment.end) / duration) * w;
    ctx.fillStyle = musicColors[label] || musicColors.unknown;
    ctx.fillRect(x, top, Math.max(2, endX - x), h);
  });
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "18px Segoe UI, sans-serif";
  ctx.fillText("音乐结构", left, top - 8);
  const progressX = left + (time / Math.max(0.001, duration)) * w;
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(progressX, top - 6, 3, h + 12);
}

function drawWaveform(ctx, width, height, duration, time) {
  const values = state.waveform;
  const left = 52;
  const top = height - 178;
  const w = width - 104;
  const h = 86;
  ctx.strokeStyle = "#334155";
  ctx.strokeRect(left, top, w, h);
  if (!values.length) return;
  ctx.fillStyle = "#3b82a8";
  values.forEach((value, index) => {
    const x = left + (index / values.length) * w;
    const barH = value * h;
    ctx.fillRect(x, top + (h - barH) / 2, Math.max(2, w / values.length - 1), barH);
  });
  const progressX = left + (time / Math.max(0.001, duration)) * w;
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(progressX, top - 8, 3, h + 16);
}

function drawRoleBands(ctx, width, height, duration, time) {
  const timeline = state.result?.timeline || [];
  const spans = state.result?.call_spans || [];
  const left = 52;
  const top = height - 62;
  const w = width - 104;
  const h = 18;
  const safeDuration = Math.max(0.001, duration);
  ctx.fillStyle = roleColors.keepspace;
  ctx.fillRect(left, top, w, h);
  if (timeline.length) {
    const points = new Set([0, safeDuration]);
    timeline.forEach((item) => {
      const start = Math.max(0, Math.min(safeDuration, Number(item.start) || 0));
      const end = Math.max(0, Math.min(safeDuration, Number(item.end) || start));
      points.add(start);
      points.add(end);
    });
    const boundaries = [...points].sort((a, b) => a - b);
    for (let index = 0; index < boundaries.length - 1; index += 1) {
      const start = boundaries[index];
      const end = boundaries[index + 1];
      if (end <= start) continue;
      const active = actionAtFromTimeline((start + end) / 2, timeline);
      const role = active?.role || "keepspace";
      const x = left + (start / safeDuration) * w;
      const endX = left + (end / safeDuration) * w;
      ctx.fillStyle = roleColors[role] || roleColors.keepspace;
      ctx.fillRect(x, top, Math.max(2, endX - x), h);
    }
  } else {
    spans.forEach((span) => {
      const x = left + (Number(span.start) / safeDuration) * w;
      const endX = left + (Number(span.end) / safeDuration) * w;
      ctx.fillStyle = roleColors[span.call_role] || roleColors.keepspace;
      ctx.fillRect(x, top, Math.max(2, endX - x), h);
    });
  }
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "18px Segoe UI, sans-serif";
  ctx.fillText("call role", left, top - 8);
  ctx.fillStyle = "#f8fafc";
  const progressX = left + (time / safeDuration) * w;
  ctx.fillRect(progressX, top - 8, 3, h + 16);
}

// ─── Downbeat Snapping ───────────────────────────────────────────────

function getDownbeats() {
  const meta = state.result?.model_meta || {};
  const raw = meta.downbeats || [];
  const anchors = raw.map(Number).filter((d) => !isNaN(d) && d >= 0);
  if (!anchors.length) {
    const generated = generatedDownbeats();
    anchors.push(...generated);
    if (!anchors.length) {
      anchors.push(0);
    }
  }
  return Array.from(new Set(anchors.map((value) => Math.round(value * 100) / 100)))
    .sort((a, b) => a - b);
}

function inferredBarSeconds() {
  const song = state.result?.song || {};
  const meta = state.result?.model_meta || {};
  const tempo = Number(song.tempo || meta.tempo);
  if (Number.isFinite(tempo) && tempo > 20 && tempo < 260) {
    return 240 / tempo;
  }

  const duration = Number(song.duration || meta.duration_s);
  const bars = Number(song.bar_count || meta.bar_count);
  if (Number.isFinite(duration) && duration > 0 && Number.isFinite(bars) && bars > 0) {
    return duration / bars;
  }

  const lengths = [];
  for (const item of state.result?.timeline || []) {
    const start = Number(item.start);
    const end = Number(item.end);
    const length = end - start;
    if (Number.isFinite(length) && length > 0.8 && length < 8) lengths.push(length);
  }
  lengths.sort((a, b) => a - b);
  return lengths.length ? lengths[Math.floor(lengths.length / 2)] : 2.4;
}

function generatedDownbeats() {
  const barSeconds = inferredBarSeconds();
  if (!Number.isFinite(barSeconds) || barSeconds <= 0) return [];
  const song = state.result?.song || {};
  const duration = Number(song.duration || state.result?.model_meta?.duration_s || 0);
  const timeline = state.result?.timeline || [];
  const firstActionStart = timeline
    .map((item) => Number(item.start))
    .find((value) => Number.isFinite(value) && value >= 0);
  const anchor = Number.isFinite(firstActionStart) ? firstActionStart : 0;
  const start = anchor - Math.ceil(anchor / barSeconds) * barSeconds;
  const end = Number.isFinite(duration) && duration > 0
    ? duration + barSeconds
    : Math.max(...timeline.map((item) => Number(item.end) || 0), anchor) + barSeconds;
  const beats = [];
  for (let t = start; t <= end + 0.001; t += barSeconds) {
    if (t >= -0.001) beats.push(Math.max(0, t));
  }
  return beats;
}

function nearestDownbeatIndex(value, downbeats) {
  let bestIndex = 0;
  let bestDist = Infinity;
  downbeats.forEach((downbeat, index) => {
    const dist = Math.abs(Number(downbeat) - value);
    if (dist < bestDist) {
      bestDist = dist;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function snapToDownbeat(value, downbeats) {
  if (!downbeats || downbeats.length === 0) return value;
  let best = downbeats[0];
  let bestDist = Math.abs(downbeats[0] - value);
  for (const db of downbeats) {
    const dist = Math.abs(db - value);
    if (dist < bestDist) {
      bestDist = dist;
      best = db;
    }
  }
  return Math.round(best * 100) / 100;
}

let _snapToastTimer = null;
function showSnapToast(original, snapped) {
  if (Math.abs(original - snapped) < 0.01) return;
  if (_snapToastTimer) clearTimeout(_snapToastTimer);
  setStatus(`重拍吸附 ${original.toFixed(2)}s → ${snapped.toFixed(2)}s`);
  _snapToastTimer = setTimeout(() => { setStatus("就绪"); }, 3000);
}

// ─── Structure Editor ────────────────────────────────────────────────

const COARSE_LABELS = [
  "intro",
  "verse",
  "pre_chorus",
  "chorus",
  "instrumental",
  "bridge",
  "outro",
];

function structureLabelOptions(currentLabel) {
  const current = String(currentLabel || "").trim();
  const labels = COARSE_LABELS.slice();
  if (current && current !== "unknown" && !labels.includes(current)) {
    labels.push(current);
  }
  return labels.map((label) => {
    const suffix = COARSE_LABELS.includes(label) ? "" : "（当前细分）";
    return `<option value="${escapeAttr(label)}" ${current === label ? "selected" : ""}>${escapeHtml(musicLabelText(label) + suffix)}</option>`;
  }).join("");
}

function renderStructureEditor() {
  const segments = state.result?.music_segments || [];
  els.segmentList.innerHTML = "";
  const hasSegments = segments.length > 0;
  els.rightPanel.hidden = !hasSegments;
  if (!hasSegments) return;

  segments.forEach((seg, index) => {
    const li = document.createElement("li");
    li.className = "segment-row";
    li.dataset.index = String(index);
    const edited = seg.source === "human_curated";
    const labelOptions = structureLabelOptions(seg.music_label);
    li.innerHTML = `
      <span class="seg-num">#${index + 1}</span>
      <input class="seg-time" type="text" value="${fmtTime(seg.start)}" data-index="${index}" data-field="start" size="8" />
      <span class="seg-dash">-</span>
      <input class="seg-time" type="text" value="${fmtTime(seg.end)}" data-index="${index}" data-field="end" size="8" />
      <select class="seg-label" data-index="${index}">${labelOptions}</select>
      <button class="seg-remove" data-index="${index}" title="Remove segment">&times;</button>
      ${edited ? '<span class="seg-edited" title="Human curated">✎</span>' : ""}
    `;
    els.segmentList.appendChild(li);
  });
  updateEditBadge();
}

function updateEditBadge() {
  const segments = state.result?.music_segments || [];
  const hasEdits = segments.some((s) => s.source === "human_curated");
  els.editBadge.textContent = hasEdits ? "已编辑" : "自动";
  els.editBadge.dataset.status = hasEdits ? "current_model" : "idle";
}

function bindStructureEvents() {
  els.segmentList.addEventListener("change", (e) => {
    const sel = e.target.closest(".seg-label");
    if (!sel) return;
    const index = Number(sel.dataset.index);
    state.result.music_segments[index].music_label = sel.value;
    state.result.music_segments[index].struct_label = sel.value;
    state.result.music_segments[index].source = "human_curated";
    state.segmentDirty = true;
    if (syncSegmentsToTimeline()) state.timelineDirty = true;
    renderStructureEditor();
    renderTimeline();
    drawCanvas();
    updateEditBadge();
  });

  els.segmentList.addEventListener("input", (e) => {
    const inp = e.target.closest(".seg-time");
    if (!inp) return;
    const index = Number(inp.dataset.index);
    const field = inp.dataset.field;
    const val = parseTimeInput(inp.value);
    if (val !== null) {
      state.result.music_segments[index][field] = Math.round(val * 100) / 100;
      state.result.music_segments[index].source = "human_curated";
      state.segmentDirty = true;
      if (syncSegmentsToTimeline()) state.timelineDirty = true;
      renderTimeline();
      drawCanvas();
      updateEditBadge();
    }
  });

  els.segmentList.addEventListener("change", (e) => {
    const inp = e.target.closest(".seg-time");
    if (!inp) return;
    const index = Number(inp.dataset.index);
    const field = inp.dataset.field;
    const downbeats = getDownbeats();
    if (downbeats.length) {
      const original = state.result.music_segments[index][field];
      const snapped = snapToDownbeat(original, downbeats);
      state.result.music_segments[index][field] = snapped;
      inp.value = fmtTime(snapped);
      showSnapToast(original, snapped);
    }
    state.result.music_segments[index].source = "human_curated";
    state.segmentDirty = true;
    if (syncSegmentsToTimeline()) state.timelineDirty = true;
    recalcAllBarCounts();
    renderTimeline();
    drawCanvas();
    updateEditBadge();
  });

  els.segmentList.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-remove");
    if (!btn) return;
    const index = Number(btn.dataset.index);
    state.result.music_segments.splice(index, 1);
    state.segmentDirty = true;
    if (syncSegmentsToTimeline()) state.timelineDirty = true;
    renderStructureEditor();
    renderTimeline();
    drawCanvas();
  });

  els.addSegmentBtn.addEventListener("click", () => {
    const segments = state.result.music_segments;
    const last = segments[segments.length - 1] || { end: 0 };
    const duration = state.result?.song?.duration || 180;
    const newStart = last.end || 0;
    const newEnd = Math.min(newStart + 16, duration);
    segments.push({
      start: Math.round(newStart * 100) / 100,
      end: Math.round(newEnd * 100) / 100,
      music_label: "verse",
      struct_label: "verse",
      source: "human_curated",
    });
    state.segmentDirty = true;
    if (syncSegmentsToTimeline()) state.timelineDirty = true;
    renderStructureEditor();
    renderTimeline();
    drawCanvas();
  });
}

// ─── Notes Editor ────────────────────────────────────────────────────

function currentNote(time) {
  const notes = state.result?.notes;
  if (!Array.isArray(notes) || !notes.length) return null;
  for (const n of notes) {
    if (time >= (n.start || 0) && time < (n.end || Infinity)) return n;
  }
  return null;
}

function renderNotesEditor() {
  const notes = state.result?.notes;
  if (!Array.isArray(notes)) return;
  els.notesList.innerHTML = "";

  notes.forEach((note, index) => {
    const li = document.createElement("li");
    li.className = "note-row";
    li.dataset.index = String(index);
    li.innerHTML = `
      <span class="seg-num">#${index + 1}</span>
      <input class="seg-time" type="text" value="${fmtTime(note.start)}" data-index="${index}" data-field="start" size="8" />
      <span class="seg-dash">-</span>
      <input class="seg-time" type="text" value="${fmtTime(note.end)}" data-index="${index}" data-field="end" size="8" />
      <input class="note-text" type="text" value="${escapeHtml(note.text || "")}" data-index="${index}" placeholder="备注内容" />
      <button class="seg-remove" data-index="${index}" title="删除备注">&times;</button>
    `;
    els.notesList.appendChild(li);
  });

  els.notesBadge.style.display = notes.length ? "inline-block" : "none";
}

function bindNotesEvents() {
  els.notesList.addEventListener("input", (e) => {
    const inp = e.target.closest(".seg-time");
    if (inp) {
      const index = Number(inp.dataset.index);
      const field = inp.dataset.field;
      const val = parseTimeInput(inp.value);
      if (val !== null) {
        state.result.notes[index][field] = Math.round(val * 100) / 100;
        state.notesDirty = true;
        drawCanvas();
      }
      return;
    }
    const textInp = e.target.closest(".note-text");
    if (textInp) {
      const index = Number(textInp.dataset.index);
      state.result.notes[index].text = textInp.value;
      state.notesDirty = true;
      drawCanvas();
    }
  });

  els.notesList.addEventListener("change", (e) => {
    const inp = e.target.closest(".seg-time");
    if (!inp) return;
    const index = Number(inp.dataset.index);
    const field = inp.dataset.field;
    const downbeats = getDownbeats();
    if (downbeats.length) {
      const original = state.result.notes[index][field];
      const snapped = snapToDownbeat(original, downbeats);
      state.result.notes[index][field] = snapped;
      inp.value = fmtTime(snapped);
      showSnapToast(original, snapped);
      state.notesDirty = true;
      drawCanvas();
    }
  });

  els.notesList.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-remove");
    if (!btn) return;
    const index = Number(btn.dataset.index);
    state.result.notes.splice(index, 1);
    state.notesDirty = true;
    renderNotesEditor();
  });

  els.addNoteBtn.addEventListener("click", () => {
    if (!Array.isArray(state.result.notes)) state.result.notes = [];
    const notes = state.result.notes;
    const last = notes[notes.length - 1];
    const duration = state.result?.song?.duration || 180;
    const newStart = last?.end || 0;
    const newEnd = Math.min(newStart + 16, duration);
    notes.push({
      start: Math.round(newStart * 100) / 100,
      end: Math.round(newEnd * 100) / 100,
      text: "",
    });
    state.notesDirty = true;
    renderNotesEditor();
  });
}

function syncSegmentsToTimeline() {
  const segments = state.result?.music_segments || [];
  const timeline = state.result?.timeline || [];
  let changed = false;
  timeline.forEach((item) => {
    const start = Number(item.start);
    const end = Number(item.end);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;

    let best = null;
    let bestOverlap = 0;
    for (const seg of segments) {
      const segStart = Number(seg.start);
      const segEnd = Number(seg.end);
      if (!Number.isFinite(segStart) || !Number.isFinite(segEnd) || segEnd <= segStart) continue;
      const overlap = Math.min(end, segEnd) - Math.max(start, segStart);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        best = seg;
      }
    }
    if (!best && segments.length) {
      const midpoint = (start + end) / 2;
      best = segments.find((seg) => midpoint >= Number(seg.start) && midpoint < Number(seg.end)) || null;
      if (best) bestOverlap = end - start;
    }
    if (!best || bestOverlap <= 0) return;
    const label = best.music_label || "unknown";
    if (item.music_label !== label || item.struct_label !== label) {
      item.music_label = label;
      item.struct_label = label;
      item.mode = "human_curated";
      changed = true;
    }
  });
  return changed;
}

// ─── Action Search ───────────────────────────────────────────────────

async function loadActionLibrary() {
  try {
    const data = await fetchJson(apiUrl("/api/actions"));
    state.actionLibrary = data.actions || [];
  } catch (_error) {
    state.actionLibrary = [];
  }
}

function filterActions(query) {
  const q = String(query || "").toLowerCase().trim();
  if (!q) return state.actionLibrary.slice(0, 20);
  return state.actionLibrary
    .filter((a) => {
      const name = String(a.display_name || a.id || "").toLowerCase();
      const cat = String(a.category || "").toLowerCase();
      const id = String(a.id || "").toLowerCase();
      const aliases = Array.isArray(a.aliases) ? a.aliases.join(" ").toLowerCase() : "";
      return name.includes(q) || cat.includes(q) || id.includes(q) || aliases.includes(q);
    })
    .slice(0, 12);
}

function normalizeActionLookup(value) {
  return String(value || "").trim().toLowerCase();
}

function findExactAction(value) {
  const q = normalizeActionLookup(value);
  if (!q) return null;
  return state.actionLibrary.find((action) => {
    const candidates = [
      action.id,
      action.display_name,
      ...(Array.isArray(action.aliases) ? action.aliases : []),
    ];
    return candidates.some((candidate) => normalizeActionLookup(candidate) === q);
  }) || null;
}

function applyActionToTimeline(index, action) {
  const item = state.result?.timeline?.[index];
  if (!item || !action) return;
  item.action_id = action.id;
  item.display_name = action.display_name || action.id;
  item.role = action.category || item.role || "keepspace";
  item.risk = action.risk || item.risk || "medium";
  item.typical_text = action.typical_text || "";
  item.tutorial_text = action.tutorial_text || null;
  recalcBarCount(item);
  item.mode = "human_curated";
}

function bindActionSearchInputs() {
  els.timelineBody.querySelectorAll(".action-search").forEach((input) => {
    input.removeEventListener("input", handleActionSearchInput);
    input.removeEventListener("focus", handleActionSearchFocus);
    input.removeEventListener("keydown", handleActionSearchKey);
    input.addEventListener("input", handleActionSearchInput);
    input.addEventListener("focus", handleActionSearchFocus);
    input.addEventListener("keydown", handleActionSearchKey);
  });
}

function handleActionSearchInput(e) {
  const input = e.target;
  const dropdown = input.nextElementSibling;
  if (!dropdown || !dropdown.classList.contains("action-dropdown")) return;
  const actions = filterActions(input.value);
  dropdown.innerHTML = actions
    .map(
      (a) => `<li data-id="${escapeAttr(a.id)}" data-name="${escapeAttr(a.display_name)}" data-cat="${escapeAttr(a.category)}" data-risk="${escapeAttr(a.risk)}" data-text="${escapeAttr(a.typical_text || "")}">
      <span class="ac-cat ac-cat-${a.category}">${a.category}</span> ${escapeHtml(a.display_name)} <span class="ac-risk">${a.risk}</span></li>`
    )
    .join("");
  dropdown.hidden = !actions.length;
}

function handleActionSearchFocus(e) {
  handleActionSearchInput(e);
}

function handleActionSearchKey(e) {
  const input = e.target;
  const dropdown = input.nextElementSibling;
  if (!dropdown) return;
  if (e.key === "Escape") {
    dropdown.hidden = true;
    input.blur();
    return;
  }
  if (e.key === "ArrowDown") {
    e.preventDefault();
    const first = dropdown.querySelector("li");
    if (first) first.focus();
    return;
  }
  if (e.key === "Enter") {
    const selected = dropdown.querySelector("li:focus, li:hover");
    if (selected) {
      e.preventDefault();
      applyActionSelection(input, selected);
    }
  }
}

document.addEventListener("click", (e) => {
  if (!e.target.closest(".action-cell")) {
    document.querySelectorAll(".action-dropdown").forEach((d) => (d.hidden = true));
  }
});

document.addEventListener("click", (e) => {
  const li = e.target.closest(".action-dropdown li");
  if (!li) return;
  const input = li.closest(".action-cell")?.querySelector(".action-search");
  if (!input) return;
  applyActionSelection(input, li);
});

function applyActionSelection(input, li) {
  const index = Number(input.dataset.index);
  const actionId = li.dataset.id;
  const action = state.actionLibrary.find((a) => a.id === actionId);
  applyActionToTimeline(index, action || {
    id: actionId,
    display_name: li.dataset.name,
    category: li.dataset.cat,
    risk: li.dataset.risk,
    typical_text: li.dataset.text,
  });
  state.timelineDirty = true;

  input.value = state.result.timeline[index].display_name;
  const dropdown = input.nextElementSibling;
  if (dropdown) dropdown.hidden = true;
  renderTimeline();
}

// ─── Save Handler ────────────────────────────────────────────────────

async function saveEdits() {
  if (!state.result) return;
  const downbeats = getDownbeats();
  if (downbeats.length) {
    for (const seg of state.result.music_segments || []) {
      seg.start = snapToDownbeat(Number(seg.start), downbeats);
      seg.end = snapToDownbeat(Number(seg.end), downbeats);
    }
    for (const note of state.result.notes || []) {
      note.start = snapToDownbeat(Number(note.start), downbeats);
      note.end = snapToDownbeat(Number(note.end), downbeats);
    }
  }
  if (syncSegmentsToTimeline()) state.timelineDirty = true;
  for (const item of state.result.timeline || []) {
    item.time = `${fmtTime(item.start)}-${fmtTime(item.end)}`;
  }
  renderStructureEditor();
  renderNotesEditor();
  const jobId = state.result.job_id || state.result?.song?.song_id || "unknown";
  const payload = {
    music_segments: state.result.music_segments || [],
    timeline: state.result.timeline || [],
    notes: state.result.notes || [],
  };
  els.saveBtn.disabled = true;
  els.saveNotesBtn.disabled = true;
  setStatus("正在保存...");
  try {
    const data = await fetchJson(apiUrl(`/api/jobs/${jobId}/save`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.result = data;
    state.segmentDirty = false;
    state.timelineDirty = false;
    state.notesDirty = false;
    if (typeof data.notes === "string") {
      state.result.notes = data.notes ? [{ start: 0, end: data.song?.duration || 180, text: data.notes }] : [];
    } else {
      state.result.notes = data.notes || [];
    }
    renderNotesEditor();
    renderStructureEditor();
    renderTimeline();
    setStatus("已保存 ✓");
    els.saveBtn.textContent = "已保存 ✓";
    els.saveNotesBtn.textContent = "已保存 ✓";
    setTimeout(() => {
      els.saveBtn.textContent = "保存编辑";
      els.saveNotesBtn.textContent = "保存备注";
    }, 2000);
  } catch (error) {
    setStatus(`保存失败: ${error.message}`);
  } finally {
    els.saveBtn.disabled = false;
    els.saveNotesBtn.disabled = false;
  }
}

// ─── Role Filter ─────────────────────────────────────────────────────

function bindRoleFilter() {
  els.timelineFilters.querySelectorAll(".filter-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      els.timelineFilters.querySelectorAll(".filter-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      els.timelineFilters.dataset.activeRole = chip.dataset.role;
      renderTimeline();
    });
  });
}

function setResult(result, audioUrl) {
  state.result = result;
  state.segmentDirty = false;
  state.timelineDirty = false;
  state.notesDirty = false;
  // Backward compat: old string notes → array
  if (typeof result.notes === "string") {
    state.result.notes = result.notes ? [{ start: 0, end: result.song?.duration || 180, text: result.notes }] : [];
  } else if (!Array.isArray(result.notes)) {
    state.result.notes = [];
  }
  recalcAllBarCounts();
  els.geiVideoOverlay.hidden = true;
  els.geiVideoPlayer.pause();
  els.geiVideoPlayer.style.display = "none";
  els.geiVideoPlayer.removeAttribute("src");
  if (els.songVideoPlayer && !state.songVideoUrl) {
    els.songVideoPlayer.pause();
    els.songVideoPlayer.style.display = "none";
  }
  stopCallAudio({ clearSource: true });
  renderStats();
  renderProcess();
  renderStructureEditor();
  renderNotesEditor();
  renderTimeline();
  els.exportJsonBtn.disabled = false;
  els.exportMdBtn.disabled = false;
  els.saveBtn.disabled = false;
  els.saveBtn.textContent = "保存编辑";
  els.saveNotesBtn.disabled = false;
  els.saveNotesBtn.textContent = "保存备注";
  els.timelineFilters.hidden = false;
  if (audioUrl) {
    els.audio.src = audioUrl;
    buildWaveformFromUrl(audioUrl);
  }
  updateExportAvailability();
  if (state.animationId) cancelAnimationFrame(state.animationId);
  drawCanvas();
}

async function loadExamples() {
  let data;
  try {
    data = await fetchJson(apiUrl("/api/songs"));
    setStatus("示例已就绪");
  } catch (_apiError) {
    try {
      data = await fetchJson("./examples/index.json");
      setStatus("静态示例就绪");
    } catch (_relativeError) {
      try {
        data = await fetchJson("/examples/index.json");
        setStatus("静态示例就绪");
      } catch (_staticError) {
        setStatus("无可用示例");
        return;
      }
    }
  }
  for (const song of data.songs || []) {
    const option = document.createElement("option");
    option.value = song.song_id;
    option.textContent = song.title;
    els.exampleSelect.appendChild(option);
  }
}

async function analyzeUpload(event) {
  event.preventDefault();
  const file = state.selectedFile || els.audioInput.files[0];
  if (!file) {
    setStatus("未选择音频");
    return;
  }
  const form = new FormData();
  form.append("audio", file);
  form.append("title", els.titleInput.value || file.name.replace(/\.[^.]+$/, ""));
  els.analyzeBtn.disabled = true;
  setStatus("正在上传...");
  try {
    const response = await fetch(apiUrl("/api/analyze"), { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.message || data.error || "analysis failed");
    const jobId = data.job_id;
    if (data.status === "done") {
      // Fast path: analysis completed synchronously
      setResult(data.result || data, data.result?.audio_url || data.audio_url);
      setStatus("分析完成");
      els.analyzeBtn.disabled = false;
      return;
    }
    // Poll for completion
    setStatus("正在分析...");
    let fatalPollError = null;
    const pollInterval = 3000;
    const maxPolls = 600; // 30 minutes max; first MuQ download can be slow.
    for (let i = 0; i < maxPolls; i++) {
      await new Promise(r => setTimeout(r, pollInterval));
      try {
        const statusResp = await fetch(apiUrl(`/api/jobs/${jobId}/status`));
        const statusData = await statusResp.json();
        if (statusData.status === "done") {
          const result = statusData.result;
          setResult(result, result.audio_url);
          setStatus("分析完成");
          els.analyzeBtn.disabled = false;
          return;
        }
        if (statusData.status === "error") {
          fatalPollError = statusData.result?.error || "analysis failed";
          throw new Error(statusData.result?.error || "分析失败");
        }
        setStatus(`正在分析... (${Math.round((i + 1) * pollInterval / 1000)}s)`);
      } catch (pollErr) {
        if (fatalPollError) throw new Error(fatalPollError);
        // Ignore poll errors, keep retrying
      }
    }
    throw new Error("分析超时，请重试");
  } catch (error) {
    const backendHint = API_BASE ? "请查看后端日志" : "需要连接 YesTiger 后端 API";
    setStatus(`错误: ${error.message}。${backendHint}`);
  } finally {
    els.analyzeBtn.disabled = false;
  }
}

async function loadExample() {
  const songId = els.exampleSelect.value;
  if (!songId) return;
  clearSongVideo({ silent: true });
  setStatus("正在加载示例...");
  try {
    const data = await fetchJson(apiUrl(`/api/examples/${songId}`));
    setResult(data, data.audio_url);
    setStatus("示例就绪");
  } catch (_error) {
    try {
      const data = await fetchJson(`./examples/${songId}.json`);
      setResult(data, data.audio_url);
      setStatus("静态示例就绪");
    } catch (_relativeError) {
      try {
        const data = await fetchJson(`/examples/${songId}.json`);
        setResult(data, data.audio_url);
        setStatus("静态示例就绪");
      } catch (error) {
        setStatus(`错误: ${error.message}`);
      }
    }
  }
}

function bindEvents() {
  els.form.addEventListener("submit", analyzeUpload);
  els.loadExampleBtn.addEventListener("click", loadExample);
  els.saveBtn.addEventListener("click", saveEdits);
  els.saveNotesBtn.addEventListener("click", saveEdits);
  bindNotesEvents();
  bindStructureEvents();
  bindRoleFilter();
  ["loadedmetadata", "canplay", "emptied", "error"].forEach((name) => {
    els.audio.addEventListener(name, () => updateExportAvailability());
  });

  ["play", "playing", "seeked", "ratechange"].forEach((name) => {
    els.audio.addEventListener(name, () => syncSongVideoToAudio(true));
  });

  ["pause", "ended"].forEach((name) => {
    els.audio.addEventListener(name, pauseOverlayVideos);
  });

  els.audioInput.addEventListener("change", async () => {
    const file = els.audioInput.files[0];
    if (!file) return;
    clearSongVideo({ silent: true });
    state.selectedFile = file;
    els.fileName.textContent = file.name;
    if (!els.titleInput.value) els.titleInput.value = file.name.replace(/\.[^.]+$/, "");
    const url = URL.createObjectURL(file);
    els.audio.src = url;
    updateExportAvailability();
    await buildWaveform(await file.arrayBuffer());
  });

  els.songVideoInput.addEventListener("change", () => {
    const file = els.songVideoInput.files[0];
    if (file) setSongVideoFile(file);
  });

  els.clearSongVideoBtn.addEventListener("click", () => clearSongVideo());

  ["dragenter", "dragover"].forEach((name) => {
    els.dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      els.dropZone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    els.dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      els.dropZone.classList.remove("is-dragging");
    });
  });
  els.dropZone.addEventListener("drop", async (event) => {
    const file = event.dataTransfer.files[0];
    if (!file) return;
    clearSongVideo({ silent: true });
    state.selectedFile = file;
    els.fileName.textContent = file.name;
    els.titleInput.value = file.name.replace(/\.[^.]+$/, "");
    const url = URL.createObjectURL(file);
    els.audio.src = url;
    updateExportAvailability();
    await buildWaveform(await file.arrayBuffer());
  });

  els.timelineBody.addEventListener("input", (event) => {
    const input = event.target.closest("[data-field]");
    if (!input || !state.result) return;
    if (input.classList.contains("time-inline")) return;
    const row = input.closest("tr");
    const index = Number(row.dataset.index);
    const field = input.dataset.field;
    if (state.result.timeline[index]) {
      if (input.classList.contains("action-search")) {
        const matched = findExactAction(input.value);
        if (matched) {
          applyActionToTimeline(index, matched);
        } else {
          state.result.timeline[index].display_name = input.value;
          state.result.timeline[index].action_id = "";
          state.result.timeline[index].mode = "human_curated";
        }
      } else {
        state.result.timeline[index][field] = input.value;
      }
      state.timelineDirty = true;
    }
  });

  els.timelineBody.addEventListener("change", (event) => {
    const inp = event.target.closest(".time-inline");
    if (!inp || !state.result) return;
    const index = Number(inp.dataset.index);
    const field = inp.dataset.field;
    const parsed = parseTimeInput(inp.value);
    if (parsed !== null) {
      let val = parsed;
      const downbeats = getDownbeats();
      if (downbeats.length) {
        const snapped = snapToDownbeat(val, downbeats);
        showSnapToast(val, snapped);
        val = snapped;
      }
      state.result.timeline[index][field] = Math.round(val * 100) / 100;
      state.result.timeline[index].time = `${fmtTime(state.result.timeline[index].start)}-${fmtTime(state.result.timeline[index].end)}`;
      state.result.timeline[index].mode = "human_curated";
      recalcAllBarCounts();
      state.timelineDirty = true;
      inp.value = fmtTime(val);
      renderTimeline();
    }
  });

  els.exportJsonBtn.addEventListener("click", () => {
    const result = editableResult();
    if (!result) return;
    downloadText(`${result.song.song_id || "yetiger"}.timeline.json`, JSON.stringify(result, null, 2), "application/json");
  });

  els.exportMdBtn.addEventListener("click", () => {
    const result = editableResult();
    if (!result) return;
    downloadText(`${result.song.song_id || "yetiger"}.callbook.md`, markdownFromTimeline(result), "text/markdown");
  });

  els.exportVideoBtn.addEventListener("click", exportTeachingVideo);
}

async function exportTeachingVideo() {
  const result = editableResult();
  if (!result || state.videoExporting) return;
  state.videoExporting = true;
  updateExportAvailability();
  setStatus("正在生成 MP4 教学视频...");
  let finalHint = null;
  try {
    let requestOptions;
    if (state.songVideoFile) {
      const form = new FormData();
      form.append("result", JSON.stringify(result));
      form.append("song_video", state.songVideoFile, state.songVideoFile.name || "song_video.mp4");
      requestOptions = { method: "POST", body: form };
    } else {
      requestOptions = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ result }),
      };
    }
    const response = await fetch(apiUrl("/api/export-video"), requestOptions);
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const error = await response.json();
        detail = error.message || error.error || detail;
      } catch (_error) {
        detail = await response.text();
      }
      throw new Error(detail);
    }
    const blob = await response.blob();
    const fallbackName = `${result.song?.song_id || "yetiger"}.teaching.mp4`;
    const filename = filenameFromDisposition(response.headers.get("Content-Disposition"), fallbackName);
    downloadBlob(filename, blob);
    setStatus("MP4 视频已下载");
    finalHint = state.songVideoFile
      ? "已保存所见即所得 MP4（含右上角歌曲视频、原曲音频和 call 声音）"
      : "已保存后端渲染 MP4（含原曲音频和 call 声音）";
  } catch (error) {
    setStatus(`MP4 导出失败: ${error.message}`);
    finalHint = "MP4 export failed";
  } finally {
    state.videoExporting = false;
    updateExportAvailability(finalHint);
  }
}


bindEvents();
updateExportAvailability();
updateSongVideoUi();

(async function init() {
  const ok = await wakeUpBackend();
  if (!ok) return;
  loadExamples();
  loadActionLibrary();
  drawCanvas();
})();
