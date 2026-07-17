const libraryState = {
  actions: [],
  version: "",
};

const els = {
  status: document.getElementById("libraryStatus"),
  search: document.getElementById("librarySearch"),
  category: document.getElementById("libraryCategory"),
  risk: document.getElementById("libraryRisk"),
  count: document.getElementById("libraryCount"),
  meta: document.getElementById("libraryMeta"),
  list: document.getElementById("libraryList"),
};

function apiUrl(path) {
  const base = (window.YESTIGER_API_BASE || "").replace(/\/$/, "");
  return `${base}${path}`;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.message || payload.error || detail;
    } catch (_error) {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function hasKana(value) {
  return /[\u3040-\u30ff]/.test(String(value || ""));
}

function hasHan(value) {
  return /[\u3400-\u9fff]/.test(String(value || ""));
}

function uniqueValues(values) {
  const seen = new Set();
  return values.filter((value) => {
    const normalized = String(value || "").trim();
    if (!normalized || seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  });
}

function extractJapaneseNames(action, aliases) {
  const names = aliases.filter(hasKana);
  const sourceTitle = String(action.tutorial_text?.source_title || "");
  const bracketPattern = /[\[［【]([^\]］】]+)[\]］】]/g;
  let match = bracketPattern.exec(sourceTitle);
  while (match) {
    if (hasKana(match[1])) names.push(match[1]);
    match = bracketPattern.exec(sourceTitle);
  }
  if (!names.length && hasKana(sourceTitle)) names.push(sourceTitle);
  return uniqueValues(names).join(" / ");
}

function categoryLabel(value) {
  return {
    mix: "MIX",
    rhythmcall: "节奏 Call",
    underground_gei: "地下艺",
    keepspace: "留白",
  }[value] || value || "-";
}

function riskLabel(value) {
  return {
    low: "低",
    medium: "中",
    high: "高",
  }[value] || value || "-";
}

function sourceLabel(value) {
  return value === "user_custom" ? "本地自定义" : "内置库";
}

function preferredBars(action) {
  const duration = action.duration || {};
  const preferred = Number(duration.preferred_bars);
  if (Number.isFinite(preferred) && preferred > 0) {
    return Number.isInteger(preferred) ? String(preferred) : String(preferred);
  }
  const allowed = action.requires?.allowed_bars;
  if (Array.isArray(allowed) && allowed.length) return allowed.join(" / ");
  const min = action.requires?.min_bars;
  const max = action.requires?.max_bars;
  if (min != null && max != null && min !== max) return `${min}-${max}`;
  if (min != null) return String(min);
  return "-";
}

function durationLabel(action) {
  const bars = preferredBars(action);
  if (bars === "-") return "长度未标注";
  const strict = action.duration?.strict_bars ? "固定" : "可调整";
  return `${bars} 小节 · ${strict}`;
}

function actionNames(action) {
  const aliases = Array.isArray(action.aliases) ? action.aliases : [];
  const chinese = action.chinese_alias
    || aliases.find((item) => hasHan(item) && !hasKana(item))
    || "";
  const japanese = extractJapaneseNames(action, aliases);
  return {
    main: action.display_name || action.id || "-",
    chinese,
    japanese,
  };
}

function actionLines(action) {
  const typical = String(action.typical_text || "").trim();
  if (typical) {
    return typical
      .split(/\n|\/|；|;/)
      .map((item) => item.trim())
      .filter(Boolean);
  }
  const bars = action.tutorial_text?.bars;
  if (Array.isArray(bars) && bars.length) {
    return bars.map((item) => String(item || "").trim()).filter(Boolean);
  }
  return action.category === "keepspace" ? ["留白 / 不喊"] : ["未填写喊词"];
}

function contextText(action) {
  const contexts = Array.isArray(action.best_context) ? action.best_context : [];
  return contexts.length ? contexts.join(" / ") : "-";
}

function searchableText(action) {
  const names = actionNames(action);
  return [
    action.id,
    names.main,
    names.chinese,
    names.japanese,
    ...(Array.isArray(action.aliases) ? action.aliases : []),
    action.typical_text,
    ...(actionLines(action)),
    contextText(action),
  ].join(" ").toLowerCase();
}

function filteredActions() {
  const query = els.search.value.trim().toLowerCase();
  const category = els.category.value;
  const risk = els.risk.value;
  return libraryState.actions.filter((action) => {
    if (category !== "all" && action.category !== category) return false;
    if (risk !== "all" && action.risk !== risk) return false;
    if (query && !searchableText(action).includes(query)) return false;
    return true;
  });
}

function renderAction(action) {
  const names = actionNames(action);
  const lines = actionLines(action);
  const sourceUrl = action.tutorial_text?.source_url;
  return `
    <article class="library-item">
      <div class="library-item-head">
        <div>
          <div class="library-title-row">
            <h2>${escapeHtml(names.chinese || names.main)}</h2>
            <span class="preview-cat preview-cat-${escapeHtml(action.category || "keepspace")}">${escapeHtml(categoryLabel(action.category))}</span>
            <span class="library-risk library-risk-${escapeHtml(action.risk || "medium")}">风险 ${escapeHtml(riskLabel(action.risk))}</span>
          </div>
          <p class="library-names">
            <span>ID: ${escapeHtml(action.id || "-")}</span>
            <span>英文/库内名: ${escapeHtml(names.main)}</span>
            <span>日文名: ${escapeHtml(names.japanese || "-")}</span>
          </p>
        </div>
        <div class="library-length">${escapeHtml(durationLabel(action))}</div>
      </div>

      <div class="library-content-grid">
        <div>
          <h3>内容</h3>
          <ol class="library-lines">
            ${lines.map((line, index) => `<li><span>${index + 1}</span>${escapeHtml(line)}</li>`).join("")}
          </ol>
        </div>
        <div>
          <h3>参考信息</h3>
          <dl class="library-facts">
            <dt>适用场景</dt>
            <dd>${escapeHtml(contextText(action))}</dd>
            <dt>来源</dt>
            <dd>${escapeHtml(sourceLabel(action.source))}${sourceUrl ? ` · <a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer">MixAndCall</a>` : ""}</dd>
            <dt>强度</dt>
            <dd>${action.intensity != null ? escapeHtml(action.intensity) : "-"}</dd>
          </dl>
        </div>
      </div>
    </article>
  `;
}

function renderLibrary() {
  const actions = filteredActions();
  els.count.textContent = String(actions.length);
  els.meta.textContent = `共 ${libraryState.actions.length} 条动作 · Library ${libraryState.version || "-"}`;
  els.list.innerHTML = actions.length
    ? actions.map(renderAction).join("")
    : `<div class="library-empty">没有匹配的 MIX / Call。</div>`;
}

async function loadLibrary() {
  els.status.textContent = "正在加载 MIX & Call 动作库...";
  try {
    const payload = await fetchJson(apiUrl("/api/library"));
    if (payload.status === "loading") {
      els.status.textContent = "后端正在预热动作库，请稍候...";
      window.setTimeout(loadLibrary, 800);
      return;
    }
    libraryState.version = payload.version || "";
    libraryState.actions = Array.isArray(payload.actions) ? payload.actions : [];
    els.status.textContent = "可按名称、喊词、类别或风险检索现有 MIX / Call。";
    renderLibrary();
  } catch (error) {
    els.status.textContent = `动作库加载失败：${error.message}`;
    els.list.innerHTML = `<div class="library-empty">动作库加载失败。</div>`;
  }
}

["input", "change"].forEach((eventName) => {
  els.search.addEventListener(eventName, renderLibrary);
  els.category.addEventListener(eventName, renderLibrary);
  els.risk.addEventListener(eventName, renderLibrary);
});

loadLibrary();
