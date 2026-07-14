// YesTiger MIX Builder — 前端逻辑
(function () {
  const els = {
    mixName: document.getElementById("mixName"),
    mixCategory: document.getElementById("mixCategory"),
    mixRisk: document.getElementById("mixRisk"),
    mixBars: document.getElementById("mixBars"),
    mixAlias: document.getElementById("mixAlias"),
    barList: document.getElementById("barList"),
    contextChips: document.getElementById("contextChips"),
    saveToLibraryBtn: document.getElementById("saveToLibraryBtn"),
    exportJsonBtn: document.getElementById("exportJsonBtn"),
    copyClipboardBtn: document.getElementById("copyClipboardBtn"),
    previewName: document.getElementById("previewName"),
    previewCategory: document.getElementById("previewCategory"),
    previewBars: document.getElementById("previewBars"),
    previewText: document.getElementById("previewText"),
    builderStatus: document.getElementById("builderStatus"),
  };

  const CONTEXT_OPTIONS = [
    "intro", "verse", "pre_chorus", "chorus", "post_chorus",
    "bridge", "solo", "instrumental_break", "outro",
    "high_tension_gap", "quiet_listening_section", "dense_vocal"
  ];

  const CONTEXT_LABELS = {
    intro: "前奏", verse: "主歌", pre_chorus: "预副歌", chorus: "副歌",
    post_chorus: "后副歌", bridge: "桥段", solo: "独奏",
    instrumental_break: "间奏", outro: "尾奏",
    high_tension_gap: "高张力空档", quiet_listening_section: "安静段落",
    dense_vocal: "密集人声"
  };

  const CATEGORY_LABELS = {
    mix: "MIX", rhythmcall: "节奏 Call", underground_gei: "地下艺", keepspace: "留白"
  };

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
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.error) {
      throw new Error(data.message || data.error || `Request failed: ${response.status}`);
    }
    return data;
  }

  function setStatus(msg) {
    els.builderStatus.textContent = msg;
  }

  function generateId(name) {
    return String(name || "custom_mix")
      .toLowerCase()
      .replace(/[^a-z0-9\u4e00-\u9fff]+/g, "_")
      .replace(/^_|_$/g, "")
      .substring(0, 60) || "custom_mix";
  }

  function renderContextChips() {
    els.contextChips.innerHTML = CONTEXT_OPTIONS.map(ctx =>
      `<label class="context-chip-label">
        <input type="checkbox" class="context-chip-check" value="${ctx}" />
        <span>${CONTEXT_LABELS[ctx] || ctx}</span>
      </label>`
    ).join("");
  }

  function getSelectedContexts() {
    return [...els.contextChips.querySelectorAll(".context-chip-check:checked")]
      .map(cb => cb.value);
  }

  function renderBarInputs() {
    const count = Math.max(1, Math.min(32, parseInt(els.mixBars.value) || 4));
    els.mixBars.value = count;
    els.barList.innerHTML = "";
    for (let i = 0; i < count; i++) {
      const li = document.createElement("li");
      li.className = "bar-row";
      li.innerHTML = `
        <span class="bar-num">Bar ${i + 1}</span>
        <input class="bar-text" type="text" placeholder="第${i + 1}小节喊词" data-bar="${i}" />
      `;
      els.barList.appendChild(li);
    }
    updatePreview();
  }

  function getBarTexts() {
    return [...els.barList.querySelectorAll(".bar-text")].map(inp => inp.value.trim());
  }

  function updatePreview() {
    const name = els.mixName.value.trim() || "未命名 MIX";
    const category = els.mixCategory.value;
    const bars = getBarTexts();

    els.previewName.textContent = name;
    els.previewCategory.textContent = CATEGORY_LABELS[category] || category;
    els.previewCategory.className = `preview-cat preview-cat-${category}`;
    els.previewBars.textContent = `${bars.filter(b => b).length}/${bars.length} 小节`;

    els.previewText.innerHTML = bars.map((t, i) =>
      `<div class="preview-bar-line"><span class="preview-bar-num">${i + 1}</span> ${t || "—"}</div>`
    ).join("");
  }

  function buildDefinition() {
    const name = els.mixName.value.trim() || "未命名 MIX";
    const category = els.mixCategory.value;
    const risk = els.mixRisk.value;
    const bars = getBarTexts().filter(b => b);
    const alias = els.mixAlias.value.trim();
    const contexts = getSelectedContexts();

    if (!bars.length) return null;

    const fullText = bars.join("\n");
    const typicalText = bars.join(" / ");

    return {
      id: generateId(name),
      display_name: name,
      aliases: alias ? [alias] : [],
      category: category,
      typical_text: typicalText,
      best_context: contexts.length ? contexts : ["verse", "chorus"],
      requires: {
        min_bars: bars.length,
        max_bars: bars.length,
        allowed_bars: [bars.length],
        vocal_density: "any",
      },
      duration: {
        preferred_bars: bars.length,
        strict_bars: true,
        can_compress: false,
        can_extend: false,
      },
      avoid: [],
      intensity: category === "keepspace" ? 0.0 : category === "rhythmcall" ? 0.5 : 0.85,
      risk: risk,
      tutorial_text: {
        source: "YesTiger MIX Builder",
        source_title: name,
        language: "romaji",
        full_text: fullText,
        bars: bars,
      },
      chinese_alias: alias || "",
    };
  }

  function exportJson() {
    const def = buildDefinition();
    if (!def) {
      setStatus("请至少填写一个小节的内容");
      return;
    }
    const json = JSON.stringify(def, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${def.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setStatus(`已导出 ${def.id}.json`);
  }

  async function copyToClipboard() {
    const def = buildDefinition();
    if (!def) {
      setStatus("请至少填写一个小节的内容");
      return;
    }
    const json = JSON.stringify(def, null, 2);
    try {
      await navigator.clipboard.writeText(json);
      setStatus("已复制到剪贴板 ✓");
    } catch {
      // Fallback
      const ta = document.createElement("textarea");
      ta.value = json;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setStatus("已复制到剪贴板 ✓");
    }
  }

  async function saveToLibrary() {
    const def = buildDefinition();
    if (!def) {
      setStatus("请至少填写一个小节的内容");
      return;
    }
    if (els.saveToLibraryBtn) els.saveToLibraryBtn.disabled = true;
    try {
      const data = await fetchJson(apiUrl("/api/custom-actions"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(def),
      });
      setStatus(`已保存到动作库: ${data.action?.display_name || def.display_name}`);
    } catch (error) {
      setStatus(`保存失败: ${error.message}`);
    } finally {
      if (els.saveToLibraryBtn) els.saveToLibraryBtn.disabled = false;
    }
  }

  function bindEvents() {
    els.mixBars.addEventListener("input", renderBarInputs);
    els.mixBars.addEventListener("change", renderBarInputs);
    els.mixName.addEventListener("input", updatePreview);
    els.mixCategory.addEventListener("change", updatePreview);
    els.barList.addEventListener("input", updatePreview);
    els.contextChips.addEventListener("change", () => {});
    if (els.saveToLibraryBtn) els.saveToLibraryBtn.addEventListener("click", saveToLibrary);
    els.exportJsonBtn.addEventListener("click", exportJson);
    els.copyClipboardBtn.addEventListener("click", copyToClipboard);
  }

  renderContextChips();
  renderBarInputs();
  bindEvents();
  setStatus("就绪 — 开始创建你的 MIX");
})();
