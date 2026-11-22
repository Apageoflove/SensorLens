"use strict";
const $ = (s) => document.querySelector(s);
const S = { sid: null, frames: [], idx: 0, cam: null, ov: null, img: null, bins: [], cats: [],
            hidden: new Set(), source: null, zoom: 1, panX: 0, panY: 0, reqId: 0, scene: null, v3d: null };
const NBINS = 48;
const canvas = $("#canvas");
const ctx = canvas.getContext("2d");

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function status(msg, cls = "") { const el = $("#status"); el.textContent = msg; el.className = cls || "muted"; }
async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = await r.text();
    try { msg = JSON.parse(msg).detail; } catch (_) { /* 非 JSON 错误体 */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r;
}
function jet(t) {
  t = Math.min(1, Math.max(0, t));
  const c = (x) => Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - x))));
  return `rgb(${c(3)},${c(2)},${c(1)})`;
}
const JET = Array.from({ length: NBINS }, (_, i) => jet(i / (NBINS - 1)));
// 柔和：jet 往白色混 40%，颜色浅、不刺眼；单色：统一浅青色，只看点的位置
const SOFT = JET.map((c) => { const [r, g, b] = c.match(/\d+/g).map(Number); const m = (x) => Math.round(x + (255 - x) * 0.4); return `rgb(${m(r)},${m(g)},${m(b)})`; });
const MONO = Array(NBINS).fill("rgb(140,235,255)");
const palette = () => ({ vivid: JET, mono: MONO }[$("#ptTone").value] || SOFT);
// 按点序号做确定性抽样，拖动滑块时点不会闪烁跳动
const keep = (i, d) => d >= 100 || ((i * 2654435761) >>> 0) % 100 < d;
const catTitle = (g) => (S.cats.find((c) => c.name === g) || {}).title || g;

// ---------------- 上传 ----------------
async function loadAdapters() {
  try {
    const list = await (await api("/api/adapters")).json();
    for (const a of list) {
      const o = document.createElement("option");
      o.value = a.name; o.textContent = a.name; o.title = a.description;
      $("#adapter").appendChild(o);
    }
  } catch (e) { status("读取适配器失败: " + e.message, "err"); }
}

// 与后端 dataset.py 的过滤规则一致：只上传可能用得上的文件，文件夹里的文档、日志、缓存、隐藏文件都不上传
const KIND = [[/\.(jpe?g|png|bmp|webp)$/i, "image"], [/\.json$/i, "json"], [/\.(bin|pcd|ply|npy)$/i, "points"],
              [/\.(zip|tar|tgz|tar\.gz)$/i, "archive"]];
const JUNK_DIR = /(^|\/)(\.[^/]*|__MACOSX|__pycache__|node_modules|venv|\$RECYCLE\.BIN|System Volume Information)\//i;
const MAX_FRAME_JSON = 64 * 1048576;
const BIG_UPLOAD = 2 * 1073741824;
const pick = { files: new Map(), junk: 0, junkBytes: 0, junkExt: {}, bigJson: [] };
const mb = (b) => b >= 1073741824 ? (b / 1073741824).toFixed(1) + " GB" : (b / 1048576).toFixed(1) + " MB";

function kindOf(path) {
  const base = path.split("/").pop();
  if (base.startsWith(".") || /^(thumbs\.db|desktop\.ini)$/i.test(base) || JUNK_DIR.test(path)) return null;
  const k = KIND.find(([re]) => re.test(base));
  return k ? k[1] : null;
}
function addFiles(list) {
  let added = 0;
  for (const { file, path } of list) {
    const p = path.replace(/\\/g, "/").replace(/^\/+/, "");
    const k = kindOf(p);
    if (!k) {
      pick.junk++; pick.junkBytes += file.size;
      const ext = (p.match(/\.[^./]+$/) || ["(无扩展名)"])[0].toLowerCase();
      pick.junkExt[ext] = (pick.junkExt[ext] || 0) + 1;
      continue;
    }
    if (k === "json" && file.size > MAX_FRAME_JSON) { pick.bigJson.push(`${p}（${mb(file.size)}）`); continue; }
    if (!pick.files.has(p)) added++;
    pick.files.set(p, { file, kind: k });
  }
  updatePickInfo();
  return added;
}
function clearPick() {
  pick.files.clear(); pick.junk = 0; pick.junkBytes = 0; pick.junkExt = {}; pick.bigJson = [];
  $("#files").value = ""; $("#folder").value = "";
  updatePickInfo();
}
function updatePickInfo() {
  const n = pick.files.size, cnt = { image: 0, json: 0, points: 0, archive: 0 };
  let bytes = 0;
  for (const { file, kind } of pick.files.values()) { cnt[kind]++; bytes += file.size; }
  $("#drop").classList.toggle("has", n > 0);
  $("#dropText").textContent = n ? `已选 ${n} 个文件，共 ${mb(bytes)}（可继续添加）` : "选择文件，或把文件 / 文件夹拖到窗口";
  $("#btnClear").hidden = !n && !pick.junk;
  const lines = [];
  if (n) lines.push(`图片 ${cnt.image} · JSON ${cnt.json} · 点云 ${cnt.points}` + (cnt.archive ? ` · 压缩包 ${cnt.archive}` : ""));
  if (n && !cnt.json && !cnt.archive) lines.push(`<span class="warn">没有 JSON：只浏览图片；有标定和标注时请一起选上</span>`);
  if (n && cnt.json && !cnt.image && !cnt.archive) lines.push(`<span class="warn">没有图片：可以先上传，再用“追加到当前”补图片</span>`);
  if (pick.junk) {
    const top = Object.entries(pick.junkExt).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k, v]) => `${k} ${v}`).join("，");
    lines.push(`已忽略 ${pick.junk} 个无关文件（${mb(pick.junkBytes)}：${esc(top)}）`);
  }
  if (pick.bigJson.length) {
    lines.push(`<span class="warn">已排除 ${pick.bigJson.length} 个超大 JSON（不像单帧标注，如 nuScenes 元数据表，` +
      `请用“打开本机目录”）：${esc(pick.bigJson.slice(0, 2).join("、"))}</span>`);
  }
  if (bytes > BIG_UPLOAD) lines.push(`<span class="warn">超过 ${mb(BIG_UPLOAD)}，上传会把文件复制一份，建议改用“打开本机目录”原地读取</span>`);
  $("#pickInfo").innerHTML = lines.join("<br>");
}

function sendFiles(url, done) {
  if (!pick.files.size) { status(pick.junk ? "选中的都是无关文件，没有可上传的数据" : "请先选择文件", "warn"); return; }
  const fd = new FormData();
  for (const [path, { file }] of pick.files) fd.append("files", file, path);
  fd.append("adapter", $("#adapter").value);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", url);
  setBusy(true);
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) $("#progress").textContent = `上传 ${mb(e.loaded)} / ${mb(e.total)}`;
  };
  xhr.upload.onload = () => { $("#progress").textContent = "上传完成，正在建立索引…"; };
  xhr.onloadend = () => {
    setBusy(false);
    let body = null;
    try { body = JSON.parse(xhr.responseText); } catch (_) { /* 非 JSON */ }
    if (xhr.status !== 200) {
      $("#progress").textContent = "";
      status("上传失败: " + (body?.detail || xhr.statusText || "网络错误"), "err");
      return;
    }
    clearPick();
    done(body);
  };
  xhr.send(fd);
}
function setBusy(b) {
  for (const id of ["#btnUpload", "#btnAppend", "#btnLocal"]) $(id).disabled = b || (id === "#btnAppend" && !canAppend());
}
const canAppend = () => !!S.sid && !S.source;
const upload = () => sendFiles("/api/sessions", onSession);
const appendFiles = () => sendFiles(`/api/sessions/${S.sid}/files`, (s) => onSession(s, true));

async function openLocal(path) {
  path = (path ?? $("#localPath").value).trim();
  if (!path) { status("请填写本机路径", "warn"); return; }
  $("#localPath").value = path;
  localStorage.setItem("adviz.localPath", path);
  setBusy(true);
  $("#progress").textContent = "正在扫描并建立索引…（nuScenes 完整数据首次约需 1 分钟，之后读缓存）";
  try {
    const r = await api("/api/sessions/local", { method: "POST", headers: { "Content-Type": "application/json" },
                                                 body: JSON.stringify({ path, adapter: $("#adapter").value }) });
    onSession(await r.json());
  } catch (e) { $("#progress").textContent = ""; status("打开失败: " + e.message, "err"); }
  setBusy(false);
}
async function rescan() {
  $("#progress").textContent = "重新扫描…";
  try { onSession(await (await api(`/api/sessions/${S.sid}/rescan`, { method: "POST" })).json(), true); }
  catch (e) { $("#progress").textContent = ""; status("重新扫描失败: " + e.message, "err"); }
}

function sessionInfo(s) {
  const ii = s.index_info || {}, out = [];
  if (s.source) out.push(`本机：<span title="${esc(s.source)}">${esc(s.source.length > 40 ? "…" + s.source.slice(-39) : s.source)}</span>`);
  if (ii.source === "nuscenes") {
    const nu = ii.nuscenes;
    out.push(`nuScenes ${esc(nu.version)}：${nu.frames} 个关键帧有数据（共 ${nu.keyframes_total}）`);
    const pres = Object.entries(nu.present).map(([k, v]) => `${k} ${v}`).join("，");
    if (pres) out.push(`有文件：${esc(pres)}`);
    if (nu.note) out.push(`<span class="warn">${esc(nu.note)}</span>`);
  }
  if (ii.image_only) out.push(`<span class="warn">没有可用的 JSON，按图片浏览（无标定、无标注）</span>`);
  if (ii.ignored_files?.count) out.push(`忽略无关文件 ${ii.ignored_files.count} 个（${mb(ii.ignored_files.bytes)}）`);
  if (ii.missing_images) {
    out.push(`<span class="warn">${ii.missing_images.frames} 帧找不到部分相机的图片，例如 ${esc(ii.missing_images.examples.slice(0, 2)
      .map(([f, c]) => `${f}: ${c.join("/")}`).join("；"))}</span>`);
  }
  if (ii.unused_images) out.push(`${ii.unused_images.count} 张图片没有被任何 JSON 引用`);
  if (s.source) out.push(`<a href="#" id="lnkRescan">重新扫描</a>`);
  return out.join("<br>");
}

function onSession(s, keepView = false) {
  const prevIdx = S.idx, prevCam = S.cam;
  S.sid = s.session_id; S.frames = s.frames; S.cats = s.categories || []; S.source = s.source || null;
  history.replaceState(null, "", "?s=" + S.sid);
  document.body.classList.toggle("has-data", s.frames.length > 0);
  setBusy(false);
  const nSeq = new Set(s.frames.map((f) => f.sequence)).size;
  const nCam = new Set(s.frames.flatMap((f) => f.cameras)).size;
  $("#progress").textContent = `${s.frames.length} 帧` + (nSeq > 1 ? `，${nSeq} 个序列` : "") + (nCam ? `，${nCam} 个相机` : "") +
    (s.skipped.length ? `，跳过 ${s.skipped.length} 个 JSON` : "") + (s.index_info?.cached ? "（索引来自缓存）" : "");
  $("#dataInfo").innerHTML = sessionInfo(s);
  $("#lnkRescan")?.addEventListener("click", (e) => { e.preventDefault(); rescan(); });
  const sel = $("#frame");
  const multiSeq = nSeq > 1;
  sel.innerHTML = s.frames.map((f, i) => `<option value="${i}">${i}: ${esc(f.frame_id)}${multiSeq && !String(f.frame_id).includes(f.sequence) ? " · " + esc(f.sequence) : ""}</option>`).join("");
  buildCats(null);
  if (s.skipped.length) {
    const rows = s.skipped.slice(0, 500);
    $("#panel").innerHTML = `<table><tr><th>跳过的文件${s.skipped.length > rows.length ? `（只列前 ${rows.length} 个，共 ${s.skipped.length}）` : ""}</th><th>原因</th></tr>${rows.map((k) =>
      `<tr><td>${esc(k.file)}</td><td class="warn">${esc(k.reason)}</td></tr>`).join("")}</table>`;
  } else $("#panel").innerHTML = "";
  if (s.frames.length) {
    if (keepView && prevIdx < s.frames.length) { S.cam = prevCam; selectFrame(prevIdx); } else selectFrame(0, true);
  } else { clearView(); status("没有可用的帧，请查看下方原因", "warn"); }
}
// ---------------- 帧 / 相机 ----------------
function selectFrame(i, fit = false) {
  if (i < 0 || i >= S.frames.length) return;
  S.idx = i; $("#frame").value = i;
  const f = S.frames[i];
  const cs = $("#camera"), prev = S.cam;
  cs.innerHTML = "";
  for (const c of f.cameras) { const o = document.createElement("option"); o.value = c; o.textContent = c; cs.appendChild(o); }
  S.cam = f.cameras.includes(prev) ? prev : f.cameras[0] || null;
  if (S.cam) cs.value = S.cam;
  loadView(fit || S.cam !== prev);
  if (!$("#view3d").hidden) load3D(String(i));
  refreshGrid();
}

function groupsParam() {
  if (!S.hidden.size) return "";
  return "&groups=" + encodeURIComponent(S.cats.map((c) => c.name).filter((n) => !S.hidden.has(n)).join(","));
}

function updateRenderLink() {
  if (!S.sid || !S.cam) return;
  const q = new URLSearchParams({ color_by: $("#colorBy").value, lidar: $("#showPts").checked, ann: $("#showAnn").checked,
                                  sync: $("#sync").checked, point_size: $("#ptSize").value,
                                  alpha: +$("#ptAlpha").value / 100, density: +$("#ptDensity").value / 100, tone: $("#ptTone").value });
  $("#lnkRender").href = `/api/sessions/${S.sid}/frames/${S.idx}/render/${encodeURIComponent(S.cam)}?${q}${groupsParam()}`;
}

function clearView(msg) {
  S.ov = null; S.img = null;
  $("#lnkRender").removeAttribute("href");
  $("#legend").innerHTML = "";
  $("#empty").hidden = false;
  resizeCanvas(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (msg) status(msg, "warn");
}

async function loadView(fit) {
  const id = ++S.reqId;
  if (!S.cam) { clearView("该帧没有相机"); return; }
  const base = `/api/sessions/${S.sid}/frames/${S.idx}`;
  const cam = encodeURIComponent(S.cam);
  const q = new URLSearchParams({ color_by: $("#colorBy").value, lidar: true, ann: true, sync: $("#sync").checked });
  updateRenderLink();
  status("加载中…");
  try {
    const [ov, img] = await Promise.all([
      api(`${base}/overlay/${cam}?${q}`).then((r) => r.json()),
      new Promise((res, rej) => {
        const im = new Image();
        im.onload = () => res(im);
        im.onerror = () => api(`${base}/image/${cam}`).then(() => rej(new Error("图片无法解码")), rej);
        im.src = `${base}/image/${cam}`;
      }),
    ]);
    if (id !== S.reqId) return;
    S.ov = ov; S.img = img;
    buildBins();
    buildCats(ov.groups || []);
    $("#empty").hidden = true;
    // 相机页隐藏时画布尺寸为 0，此时适配会得到极小的缩放；推迟到切回相机页再适配
    if (fit || S.needsFit) { if ($("#viewCam").hidden) S.needsFit = true; else { S.needsFit = false; fitView(); } }
    draw();
    const st = Object.entries(ov.stats || {}).map(([k, v]) => `${k}: ${v.in_view}/${v.total} 点在视野内`).join("，");
    const sizeWarn = (img.naturalWidth !== ov.width || img.naturalHeight !== ov.height)
      ? `；图片 ${img.naturalWidth}x${img.naturalHeight} 与内参 ${ov.width}x${ov.height} 不一致，已按比例缩放` : "";
    const n2 = ov.annotations.filter((a) => a.is2d).length;
    status(`${S.frames[S.idx].frame_id} / ${S.cam}  ${st}  标注 ${ov.annotations.length} 个（2D ${n2}，3D ${ov.annotations.length - n2}）` +
           (ov.has_intrinsic ? "" : "  [无内参，仅 2D]") + sizeWarn, sizeWarn ? "warn" : "muted");
    const msgs = ov.warnings.map((w) => `<div class="warn">${esc(w)}</div>`)
      .concat((ov.notes || []).map((n) => `<div class="muted">· ${esc(n)}</div>`));
    $("#panel").innerHTML = msgs.join("");
  } catch (e) {
    if (id === S.reqId) { clearView(); status("加载失败: " + e.message, "err"); }
  }
}

function buildBins() {
  const p = S.ov.points;
  S.bins = Array.from({ length: NBINS }, () => []);
  if (!p) { $("#legend").innerHTML = ""; return; }
  const maxD = +$("#maxDepth").value || Infinity;
  const span = p.vmax - p.vmin || 1;
  for (let i = 0; i < p.u.length; i++) {
    if (p.depth[i] > maxD) continue;
    const b = Math.max(0, Math.min(NBINS - 1, Math.floor(((p.value[i] - p.vmin) / span) * (NBINS - 1))));
    S.bins[b].push(i);
  }
  const unit = { depth: "m", intensity: "", height: "m" }[p.color_by];
  $("#legend").innerHTML = `${esc({ depth: "深度", intensity: "强度", height: "高度(自车系)" }[p.color_by])}
    <div class="bar"></div>${p.vmin.toFixed(1)}${unit} — ${p.vmax.toFixed(1)}${unit}`;
}

// ---------------- 类别筛选 ----------------
function buildCats(groups) {
  const box = $("#cats");
  // 还没拿到叠加数据时先按会话里的类别配置列出来，不必等第一帧加载完
  if (groups === null) groups = S.cats;
  if (!groups.length) { box.innerHTML = '<span class="muted">本帧没有标注</span>'; return; }
  box.innerHTML = groups.map((g) => `<label class="cat" title="${esc(g.name)}">
      <input type="checkbox" data-g="${esc(g.name)}" ${S.hidden.has(g.name) ? "" : "checked"}>
      <i style="background:${esc(g.color)}"></i>${esc(g.title)}
      <span class="muted">${g["3d"] ? "3D " + g["3d"] : ""}${g["3d"] && g["2d"] ? " · " : ""}${g["2d"] ? "2D " + g["2d"] : ""}</span></label>`).join("");
  for (const cb of box.querySelectorAll("input")) {
    cb.onchange = () => {
      if (cb.checked) S.hidden.delete(cb.dataset.g); else S.hidden.add(cb.dataset.g);
      updateRenderLink(); draw(); if (S.scene) build3DLines(); refreshGrid();
    };
  }
}
function visible(a) {
  if (S.hidden.has(a.group)) return false;
  return a.is2d ? $("#show2d").checked : $("#show3d").checked;
}

// ---------------- 绘制 ----------------
function resizeCanvas() {
  const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(r.width * dpr));
  canvas.height = Math.max(1, Math.round(r.height * dpr));
}
function fitView() {
  if (!S.img) return;
  resizeCanvas();
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width / dpr, h = canvas.height / dpr;
  S.zoom = Math.min(w / S.img.naturalWidth, h / S.img.naturalHeight);
  S.panX = (w - S.img.naturalWidth * S.zoom) / 2;
  S.panY = (h - S.img.naturalHeight * S.zoom) / 2;
}
function draw() {
  if (!S.img || !S.ov) return;
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#0e0e0e"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(dpr * S.zoom, 0, 0, dpr * S.zoom, dpr * S.panX, dpr * S.panY);
  ctx.drawImage(S.img, 0, 0);
  ctx.save();
  ctx.beginPath(); ctx.rect(0, 0, S.img.naturalWidth, S.img.naturalHeight); ctx.clip();
  const sx = S.img.naturalWidth / S.ov.width, sy = S.img.naturalHeight / S.ov.height;
  const p = S.ov.points;
  if (p && $("#showPts").checked) {
    // 点大小按原图像素计，缩小查看时点和图像一起缩小，不会连成一片盖住图像；最小 1 个屏幕像素
    const s = Math.max(+$("#ptSize").value, 1 / S.zoom), h = s / 2, dens = +$("#ptDensity").value, pal = palette();
    ctx.globalAlpha = +$("#ptAlpha").value / 100;
    for (let b = 0; b < NBINS; b++) {
      const g = S.bins[b];
      if (!g.length) continue;
      ctx.fillStyle = pal[b];
      for (const i of g) if (keep(i, dens)) ctx.fillRect((p.u[i] + 0.5) * sx - 0.5 - h, (p.v[i] + 0.5) * sy - 0.5 - h, s, s);
    }
    ctx.globalAlpha = 1;
  }
  if ($("#showAnn").checked) {
    const fs = 12 / S.zoom;
    ctx.font = `${fs}px "Cascadia Mono", Consolas, "Microsoft YaHei", monospace`;
    ctx.textBaseline = "bottom";
    const labels = [];
    const ids3d = new Set(S.ov.annotations.filter((x) => !x.is2d && x.id && visible(x)).map((x) => x.id));
    for (const a of S.ov.annotations) {
      if (!visible(a)) continue;
      const hot = a === S.hover;
      ctx.lineWidth = (hot ? 4 : 2) / S.zoom;
      ctx.strokeStyle = a.color; ctx.fillStyle = a.color;
      let anchor = null;
      for (const line of a.lines || []) {
        ctx.beginPath();
        line.forEach(([x, y], k) => (k ? ctx.lineTo(x * sx, y * sy) : ctx.moveTo(x * sx, y * sy)));
        ctx.stroke();
        for (const q of line) {
          if (q[0] < 0 || q[1] < 0 || q[0] > S.ov.width || q[1] > S.ov.height) continue;
          // 框标签放在顶部；折线标签放在离车最近（图像最下方）的一端，避免远处挤成一堆
          const isLine = a.kind === "polyline" || a.kind === "polyline2d";
          if (!anchor || (isLine ? q[1] > anchor[1] : q[1] < anchor[1])) anchor = q;
        }
      }
      for (const [x, y] of a.points || []) {
        ctx.beginPath(); ctx.arc(x * sx, y * sy, 4 / S.zoom, 0, 2 * Math.PI); ctx.stroke();
        anchor = anchor || [x, y];
      }
      if (anchor && $("#showLabel").checked && a.label && !(a.is2d && a.id && ids3d.has(a.id))) {
        labels.push({ x: anchor[0] * sx, y: anchor[1] * sy, text: a.label, color: a.color, hot });
      }
    }
    // 标签画在最上层：带底色的小标签，按 y 排序后对重叠的往上挪
    const pad = 3 / S.zoom, h = fs + 2 * pad, placed = [];
    labels.sort((p, q) => p.y - q.y);
    for (const L of labels) {
      const w = ctx.measureText(L.text).width + 2 * pad;
      let x = L.x, y = L.y - h - 2 / S.zoom;
      for (let k = 0; k < 6; k++) {
        const hit = placed.find((p) => x < p.x + p.w && x + w > p.x && y < p.y + p.h && y + h > p.y);
        if (!hit) break;
        y = hit.y - h - 1 / S.zoom;
      }
      placed.push({ x, y, w, h });
      ctx.globalAlpha = L.hot ? 1 : 0.88;
      ctx.fillStyle = L.color;
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#111";
      ctx.fillText(L.text, x + pad, y + h - pad + 1 / S.zoom);
    }
  }
  ctx.restore();
}

function segDist2(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, l = dx * dx + dy * dy;
  const t = l ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l)) : 0;
  return (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2;
}
function pickAnnotation(u, v, tol) {
  let best = null, bd = tol * tol;
  for (const a of S.ov.annotations) {
    if (!visible(a)) continue;
    for (const line of a.lines || []) {
      for (let k = 1; k < line.length; k++) {
        const d = segDist2(u, v, line[k - 1][0], line[k - 1][1], line[k][0], line[k][1]);
        if (d < bd) { bd = d; best = a; }
      }
    }
    for (const [x, y] of a.points || []) {
      const d = (x - u) ** 2 + (y - v) ** 2;
      if (d < bd) { bd = d; best = a; }
    }
  }
  return best;
}
function describe(a) {
  const attrs = Object.entries(a.attrs || {}).map(([k, v]) => `${esc(k)}: ${esc(v)}`).join("<br>");
  return `<b style="color:${esc(a.color)}">${esc(a.label || "(无标签)")}</b> <span class="muted">${esc(catTitle(a.group))} · ${esc(a.kind)}</span>` +
    (a.id ? `<br>id: ${esc(a.id)}` : "") + (attrs ? `<br>${attrs}` : "");
}

// ---------------- 交互 ----------------
let drag = null;
canvas.addEventListener("mousedown", (e) => { drag = { x: e.clientX, y: e.clientY, px: S.panX, py: S.panY }; canvas.classList.add("drag"); });
window.addEventListener("mouseup", () => { drag = null; canvas.classList.remove("drag"); });
canvas.addEventListener("mousemove", (e) => {
  if (drag) { S.panX = drag.px + e.clientX - drag.x; S.panY = drag.py + e.clientY - drag.y; draw(); }
  if (!S.ov || !S.img) return;
  const r = canvas.getBoundingClientRect();
  const k = S.ov.width / S.img.naturalWidth;
  const u = (e.clientX - r.left - S.panX) / S.zoom * k;
  const v = (e.clientY - r.top - S.panY) / S.zoom * (S.ov.height / S.img.naturalHeight);
  let txt = `u=${u.toFixed(1)} v=${v.toFixed(1)}  缩放 ${(S.zoom * 100).toFixed(0)}%`;
  const p = S.ov.points;
  if (p && p.u.length && $("#showPts").checked) {
    let best = -1, bd = (6 / S.zoom * k) ** 2;
    for (let i = 0; i < p.u.length; i++) {
      const d = (p.u[i] - u) ** 2 + (p.v[i] - v) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    if (best >= 0) txt += `  最近点 深度 ${p.depth[best].toFixed(2)} m`;
  }
  $("#hud").textContent = txt;
  const hit = !drag && $("#showAnn").checked ? pickAnnotation(u, v, 6 / S.zoom * k) : null;
  const tip = $("#tip");
  if (hit) {
    tip.innerHTML = describe(hit); tip.hidden = false;
    tip.style.left = `${e.clientX - r.left + 14}px`; tip.style.top = `${e.clientY - r.top + 14}px`;
  } else tip.hidden = true;
  if (hit !== S.hover) { S.hover = hit; draw(); }
});
canvas.addEventListener("mouseleave", () => { $("#tip").hidden = true; if (S.hover) { S.hover = null; draw(); } });
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  const k = Math.exp(-e.deltaY * 0.0015);
  S.panX = mx - (mx - S.panX) * k; S.panY = my - (my - S.panY) * k; S.zoom *= k;
  draw();
}, { passive: false });
canvas.addEventListener("dblclick", () => { fitView(); draw(); });
window.addEventListener("resize", () => { resizeCanvas(); draw(); if (S.v3d && !$("#view3d").hidden) { S.v3d.resize(); draw3DLabels(); } });
window.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  if (e.key === "ArrowLeft") selectFrame(S.idx - 1);
  if (e.key === "ArrowRight") selectFrame(S.idx + 1);
});

$("#btnUpload").onclick = upload;
$("#btnAppend").onclick = appendFiles;
$("#frame").onchange = (e) => selectFrame(+e.target.value);
$("#prev").onclick = () => selectFrame(S.idx - 1);
$("#next").onclick = () => selectFrame(S.idx + 1);
$("#camera").onchange = (e) => { S.cam = e.target.value; loadView(true); };
$("#colorBy").onchange = () => { loadView(false); refreshGrid(); };
$("#sync").onchange = () => { loadView(false); refreshGrid(); };
for (const id of ["#showPts", "#showAnn", "#show2d", "#show3d"]) $(id).onchange = () => { updateRenderLink(); draw(); if (S.scene) build3DLines(); refreshGrid(); };
for (const id of ["#showLabel", "#ptSize", "#ptAlpha", "#ptDensity", "#ptTone"]) {
  $(id).oninput = () => { updateRenderLink(); draw(); };
  $(id).addEventListener("change", refreshGrid);
}
$("#maxDepth").onchange = () => { if (S.ov) { buildBins(); draw(); } };

function refreshGrid() { if (!$("#viewGrid").hidden) loadGrid(); }
function showTab(name) {
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("active", x.dataset.tab === name));
  $("#viewCam").hidden = name !== "cam";
  $("#viewBev").hidden = name !== "bev";
  $("#view3d").hidden = name !== "3d";
  $("#viewGrid").hidden = name !== "grid";
  if (name === "grid") loadGrid();
  if (name === "cam") { if (S.needsFit) { S.needsFit = false; fitView(); } else resizeCanvas(); draw(); if (S.ov) status(`${S.frames[S.idx].frame_id} / ${S.cam}`); }
  if (name === "3d") {
    if (!S.scene || S.scene.key !== sceneKey()) load3D();
    else { S.v3d.resize(); draw3DLabels(); }
  }
}
for (const b of document.querySelectorAll("#tabs button")) b.onclick = () => showTab(b.dataset.tab);

// ---------------- 环视：当前帧所有相机 ----------------
const CAM_ORDER = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT", "LEFT", "RIGHT", "BACK_LEFT", "BACK", "BACK_RIGHT"];
function camRank(c) {
  const k = c.toUpperCase().replace(/^CAM(ERA)?[_-]?/, "").replace(/-/g, "_");
  const i = CAM_ORDER.indexOf(k);
  return i < 0 ? CAM_ORDER.length : i;
}
function loadGrid() {
  if (!S.sid || !S.frames.length) return;
  const f = S.frames[S.idx];
  const cams = [...f.cameras].sort((a, b) => camRank(a) - camRank(b));
  const cols = cams.length <= 1 ? 1 : cams.length <= 4 ? 2 : 3;
  const box = $("#grid");
  box.style.setProperty("--cols", cols);
  const w = Math.min(1600, Math.ceil(($("#viewGrid").clientWidth || 1200) / cols * (window.devicePixelRatio || 1)));
  const q = new URLSearchParams({ color_by: $("#colorBy").value, lidar: $("#showPts").checked, ann: $("#showAnn").checked,
                                  sync: $("#sync").checked, point_size: $("#ptSize").value, alpha: +$("#ptAlpha").value / 100,
                                  density: +$("#ptDensity").value / 100, tone: $("#ptTone").value, max_w: Math.max(64, w) });
  const key = `${S.sid}|${S.idx}|${q}${groupsParam()}`;
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  box.innerHTML = cams.length ? "" : '<p class="muted">该帧没有相机</p>';
  for (const c of cams) {
    const fig = document.createElement("figure");
    fig.title = `${c}：点击放大查看`;
    const img = new Image();
    img.alt = c;
    img.onerror = () => { fig.innerHTML = `<figcaption>${esc(c)}</figcaption><div class="err">无法显示：图片缺失或标定有误</div>`; };
    img.src = `/api/sessions/${S.sid}/frames/${S.idx}/render/${encodeURIComponent(c)}?${q}${groupsParam()}`;
    fig.append(img);
    fig.insertAdjacentHTML("beforeend", `<figcaption>${esc(c)}</figcaption>`);
    fig.onclick = () => { S.cam = c; $("#camera").value = c; showTab("cam"); loadView(true); };
    box.append(fig);
  }
  status(`${f.frame_id}：${cams.length} 个相机`);
}
// ---------------- 外参检查 ----------------
const VERDICT = { ok: "正常", suspect: "疑似有误", uncertain: "无法确认", unknown: "无法检查", error: "出错" };
const METHOD = { intensity_mi: "图像灰度与点云强度的互信息", depth_edges: "点云深度跳变与图像边缘的对齐度",
                 box_iou: "3D 框投影与同目标 2D 框的 IoU", annotation_edges: "3D 标注投影线与图像边缘的重合度（置信度较低）" };
$("#btnCheck").onclick = async () => {
  const btn = $("#btnCheck"); btn.disabled = true; status("外参检查中…");
  try {
    const r = await (await api(`/api/sessions/${S.sid}/frames/${S.idx}/calibcheck/${encodeURIComponent(S.cam)}?sync=${$("#sync").checked}`)).json();
    const rows = (r.hypotheses || []).slice(0, 8).map((h) => `<tr><td>${h.changes.length ? esc(h.changes.join("；")) : "<b>当前约定</b>"}</td>
      <td>${h.score}</td><td>${h.match}</td><td>${h.prior}</td><td>${h.in_view}</td><td class="warn">${esc(h.notes)}</td></tr>`).join("");
    $("#panel").innerHTML = `<p class="v-${r.verdict}">【${VERDICT[r.verdict]}】${esc(r.message)}</p>` + (r.method ?
      `<p class="muted">方法：${esc(METHOD[r.method] || r.method)} × 物理先验。列出得分前 8 的约定组合。</p>
      <table><tr><th>相对当前约定的改动</th><th>总分</th><th>图像匹配</th><th>先验</th><th>视野内数量</th><th>备注</th></tr>${rows}</table>` : "");
    status("外参检查完成");
  } catch (e) { status("外参检查失败: " + e.message, "err"); }
  btn.disabled = false;
};
$("#btnBatch").onclick = async () => {
  const btn = $("#btnBatch"); btn.disabled = true; status("批量检查中，帧数多时需要一些时间…");
  try {
    const r = await (await api(`/api/sessions/${S.sid}/batch_check?max_frames=${+$("#batchN").value || 50}`, { method: "POST" })).json();
    const sum = Object.entries(r.counts).map(([k, v]) => `<span class="v-${k}">${VERDICT[k] || k} ${v}</span>`).join("，");
    const rows = r.rows.map((x) => `<tr class="clickable" data-i="${x.index}" data-c="${esc(x.camera)}"><td>${x.index}</td><td>${esc(x.frame_id)}</td>
      <td>${esc(x.camera)}</td><td class="v-${x.verdict}">${VERDICT[x.verdict] || x.verdict}</td><td>${x.score ?? ""}</td><td>${esc(x.message)}</td></tr>`).join("");
    $("#panel").innerHTML = `<p>共 ${r.total} 项：${sum}。点击行跳转。</p>
      <table><tr><th>#</th><th>帧</th><th>相机</th><th>结论</th><th>得分</th><th>说明</th></tr>${rows}</table>`;
    for (const tr of document.querySelectorAll("#panel tr.clickable")) {
      tr.onclick = () => { S.cam = tr.dataset.c; showTab("cam"); selectFrame(+tr.dataset.i); };
    }
    status("批量检查完成");
  } catch (e) { status("批量检查失败: " + e.message, "err"); }
  btn.disabled = false;
};

// ---------------- BEV ----------------
function bevQuery() {
  const q = new URLSearchParams({ frames: $("#bevFrames").value || "0", target: $("#bevTarget").value,
    range_m: $("#bevRange").value, voxel: $("#bevVoxel").value, sync: $("#sync").checked });
  return q.toString() + groupsParam();
}
$("#btnBev").onclick = async () => {
  showTab("bev");
  const btn = $("#btnBev"); btn.disabled = true; status("生成 BEV…");
  try {
    const r = await api(`/api/sessions/${S.sid}/bev?${bevQuery()}`);
    const url = URL.createObjectURL(await r.blob());
    const old = $("#bevImg").src; $("#bevImg").src = url; if (old.startsWith("blob:")) URL.revokeObjectURL(old);
    $("#bevInfo").textContent = `${r.headers.get("X-Point-Count")} 个点。上方为车头方向（x+），左侧为 y+，圆圈间隔 10 m，白色方块为各帧自车位置。`;
    status("BEV 完成");
  } catch (e) { status("BEV 失败: " + e.message, "err"); }
  btn.disabled = false;
};
$("#lnkPly").onclick = (e) => { e.preventDefault(); location.href = `/api/sessions/${S.sid}/stitch.ply?${bevQuery()}`; };

// ---------------- 3D 视图 ----------------
const BOX_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
function sceneKey(frames) {
  return [frames ?? ($("#bevFrames").value || "0"), $("#bevTarget").value, $("#sync").checked, $("#bevVoxel").value].join("|");
}
async function load3D(frames) {
  if (!S.sid) return;
  if (frames !== undefined) $("#bevFrames").value = frames;
  if (!S.v3d) {
    try { S.v3d = new View3D($("#gl")); } catch (e) { status("3D 视图不可用: " + e.message, "err"); return; }
    S.v3d.onchange = draw3DLabels;
  }
  S.v3d.resize();
  const key = sceneKey();
  const id = S.load3dId = (S.load3dId || 0) + 1;
  const q = new URLSearchParams({ frames: $("#bevFrames").value || "0", target: $("#bevTarget").value, sync: $("#sync").checked });
  const qc = new URLSearchParams({ frames: q.get("frames"), target: q.get("target"), sync: $("#sync").checked,
                                   voxel: $("#bevVoxel").value, max_points: 800000 });
  status("加载 3D 场景…");
  try {
    const [scene, cloud] = await Promise.all([
      api(`/api/sessions/${S.sid}/scene?${q}`).then((r) => r.json()),
      api(`/api/sessions/${S.sid}/cloud.bin?${qc}`).then(async (r) => ({ buf: await r.arrayBuffer(), notes: r.headers.get("X-Notes") }))
        .catch((e) => ({ buf: new ArrayBuffer(0), notes: null, err: e.message })),
    ]);
    if (id !== S.load3dId) return;
    scene.key = key;
    S.scene = scene;
    const pts = new Float32Array(cloud.buf);
    S.v3d.setPoints(pts);
    S.v3d.setMode($("#color3d").value === "intensity" ? 1 : 0);
    build3DLines();
    const o = scene.origins;
    const c = o.reduce((s, p) => [s[0] + p[0] / o.length, s[1] + p[1] / o.length, s[2] + p[2] / o.length], [0, 0, 0]);
    const span = Math.max(30, ...o.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1]) * 2));
    S.v3d.cam.yaw = Math.PI;  // 从车后上方看
    S.v3d.frame(c, span);
    draw3DLabels();
    let notes = [];
    try { notes = cloud.notes ? JSON.parse(decodeURIComponent(cloud.notes)) : []; } catch (_) { /* 忽略 */ }
    $("#info3d").textContent = `${pts.length / 4} 个点，${scene.annotations.length} 个 3D 标注，帧 ${scene.frames.length} 个` +
      (scene.has_lidar ? "" : "（无点云，只显示标注）") + (cloud.err ? "  ·  点云加载失败: " + cloud.err : "") +
      (notes.length ? "  ·  " + notes.join("；") : "");
    status("3D 场景已加载。左键旋转，右键或 Shift+左键平移，滚轮缩放");
  } catch (e) { if (id === S.load3dId) status("3D 加载失败: " + e.message, "err"); }
}
function build3DLines() {
  if (!S.scene || !S.v3d) return;
  const segs = [];
  const show = $("#showAnn").checked && $("#show3d").checked;
  for (const a of S.scene.annotations) {
    if (!show || S.hidden.has(a.group)) continue;
    const P = a.points;
    if (a.kind === "box3d") {
      for (const [i, j] of a.edges || BOX_EDGES) segs.push({ a: P[i], b: P[j], color: a.color });
      const f = [(P[0][0] + P[1][0] + P[4][0] + P[5][0]) / 4, (P[0][1] + P[1][1] + P[4][1] + P[5][1]) / 4, (P[0][2] + P[1][2] + P[4][2] + P[5][2]) / 4];
      const ctr = P.reduce((s, p) => [s[0] + p[0] / 8, s[1] + p[1] / 8, s[2] + p[2] / 8], [0, 0, 0]);
      segs.push({ a: ctr, b: f, color: a.color });
    } else if (a.kind === "points") {
      for (const p of P) {
        for (const d of [[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]]) {
          segs.push({ a: [p[0] - d[0], p[1] - d[1], p[2] - d[2]], b: [p[0] + d[0], p[1] + d[1], p[2] + d[2]], color: a.color });
        }
      }
    } else {
      const seq = a.kind === "polygon" ? [...P, P[0]] : P;
      for (let k = 1; k < seq.length; k++) segs.push({ a: seq[k - 1], b: seq[k], color: a.color });
    }
  }
  const o = S.scene.origins;
  for (let k = 1; k < o.length; k++) segs.push({ a: o[k - 1], b: o[k], color: "#ffffff" });
  for (const p of o) segs.push({ a: [p[0], p[1], 0], b: [p[0], p[1], 2], color: "#ffffff" });
  for (const c of S.scene.cameras) {
    const p = c.position, f = c.forward;
    segs.push({ a: p, b: [p[0] + f[0] * 2, p[1] + f[1] * 2, p[2] + f[2] * 2], color: "#d9a441" });
  }
  S.v3d.setLines(segs);
  draw3DLabels();
}
function draw3DLabels() {
  const cv = $("#gl2d"), g = cv.getContext("2d"), r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, r.width, r.height);
  if (!S.scene || !$("#showLabel").checked || !$("#showAnn").checked || !$("#show3d").checked) return;
  g.font = '11px "Cascadia Mono", Consolas, "Microsoft YaHei", monospace';
  g.textBaseline = "middle";
  const items = [];
  for (const a of S.scene.annotations) {
    if (S.hidden.has(a.group) || a.kind !== "box3d") continue;
    const top = a.points.slice(4).reduce((s, p) => [s[0] + p[0] / 4, s[1] + p[1] / 4, s[2] + p[2] / 4], [0, 0, 0]);
    const q = S.v3d.project(top);
    if (q && q[0] >= 0 && q[1] >= 0 && q[0] <= r.width && q[1] <= r.height) items.push({ q, text: a.label, color: a.color });
  }
  for (const c of S.scene.cameras) {
    const q = S.v3d.project(c.position);
    if (q) items.push({ q: [q[0], q[1] + 22], text: c.name, color: "#d9a441" });
  }
  // 近处（屏幕靠下）的优先放，重叠的往上挪
  items.sort((p, q) => q.q[1] - p.q[1]);
  const placed = [], h = 17;
  for (const it of items) {
    const w = g.measureText(it.text).width + 10;
    const x = it.q[0] - w / 2;
    let y = it.q[1] - h - 4;
    for (let k = 0; k < 6; k++) {
      const hit = placed.find((p) => x < p.x + p.w && x + w > p.x && y < p.y + h && y + h > p.y);
      if (!hit) break;
      y = hit.y - h - 2;
    }
    placed.push({ x, y, w });
    g.globalAlpha = 0.9; g.fillStyle = it.color;
    g.fillRect(x, y, w, h);
    g.globalAlpha = 1; g.fillStyle = "#111"; g.fillText(it.text, x + 5, y + h / 2 + 0.5);
  }
}
$("#btn3d").onclick = () => { showTab("3d"); load3D(); };
$("#color3d").onchange = () => { if (S.v3d) S.v3d.setMode($("#color3d").value === "intensity" ? 1 : 0); };
$("#size3d").oninput = () => { if (S.v3d) { S.v3d.opts.size = +$("#size3d").value; S.v3d.draw(); } };

// ---------------- 上传区：选择、拖放（支持拖入文件夹） ----------------
const fromInput = (inp) => [...inp.files].map((file) => ({ file, path: file.webkitRelativePath || file.name }));
$("#files").onchange = () => { addFiles(fromInput($("#files"))); $("#files").value = ""; };
$("#folder").onchange = () => { addFiles(fromInput($("#folder"))); $("#folder").value = ""; };
$("#btnClear").onclick = (e) => { e.preventDefault(); clearPick(); };
async function readEntry(entry, out) {
  if (entry.isFile) {
    await new Promise((res) => entry.file((file) => { out.push({ file, path: entry.fullPath.replace(/^\//, "") }); res(); }, res));
  } else if (entry.isDirectory) {
    if (JUNK_DIR.test(entry.fullPath.replace(/^\//, "") + "/")) return;
    const reader = entry.createReader();
    // readEntries 每次最多返回 100 个左右，要循环读到空
    for (;;) {
      const batch = await new Promise((res) => reader.readEntries(res, () => res([])));
      if (!batch.length) break;
      for (const e of batch) await readEntry(e, out);
    }
  }
}
let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  if (![...(e.dataTransfer?.types || [])].includes("Files")) return;
  dragDepth++; document.body.classList.add("dragging"); $("#drop").classList.add("over");
});
window.addEventListener("dragleave", () => {
  if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dragging"); $("#drop").classList.remove("over"); }
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", async (e) => {
  e.preventDefault();
  dragDepth = 0; document.body.classList.remove("dragging"); $("#drop").classList.remove("over");
  const entries = [...(e.dataTransfer?.items || [])].map((it) => it.webkitGetAsEntry?.()).filter(Boolean);
  let list = [];
  if (entries.length) {
    status("读取拖入的文件…");
    for (const en of entries) await readEntry(en, list);
  } else {
    list = [...(e.dataTransfer?.files || [])].map((file) => ({ file, path: file.name }));
  }
  if (!list.length) { status("没有读到文件", "warn"); return; }
  showSrc("upload");
  const n = addFiles(list);
  status(`添加了 ${n} 个文件，点击“上传并新建”或“追加到当前”`);
});
function showSrc(name) {
  for (const b of document.querySelectorAll("#srcTabs button")) b.classList.toggle("active", b.dataset.src === name);
  $("#srcUpload").hidden = name !== "upload"; $("#uploadBtns").hidden = name !== "upload";
  $("#srcLocal").hidden = name !== "local"; $("#btnLocal").hidden = name !== "local";
}
for (const b of document.querySelectorAll("#srcTabs button")) b.onclick = () => showSrc(b.dataset.src);
$("#localPath").value = localStorage.getItem("adviz.localPath") || "";
$("#localPath").addEventListener("keydown", (e) => { if (e.key === "Enter") openLocal(); });
$("#btnLocal").onclick = () => openLocal();
// ---------------- 小组件 ----------------
for (const o of document.querySelectorAll("output[for]")) {
  const inp = document.getElementById(o.htmlFor.value || o.getAttribute("for"));
  const sync = () => { o.textContent = inp.value; };
  inp.addEventListener("input", sync); sync();
}
$("#sideToggle").onclick = () => {
  document.body.classList.toggle("side-collapsed");
  localStorage.setItem("adviz.side", document.body.classList.contains("side-collapsed") ? "0" : "1");
  window.dispatchEvent(new Event("resize"));
  if (S.img) { fitView(); draw(); }
};
if (localStorage.getItem("adviz.side") === "0" || (localStorage.getItem("adviz.side") === null && innerWidth < 820)) {
  document.body.classList.add("side-collapsed");
}
$("#drawerToggle").onclick = () => $("#drawer").classList.toggle("collapsed");
new MutationObserver(() => $("#drawer").classList.remove("collapsed")).observe($("#panel"), { childList: true });
// ---------------- 示例 ----------------
async function loadSamples() {
  try {
    const names = await (await api("/api/samples")).json();
    if (!names.length) return;
    $("#sampleList").innerHTML = names.map((n) => `<a href="#" data-n="${esc(n)}">${esc(n)}</a>`).join("");
    for (const a of $("#sampleList").querySelectorAll("a")) {
      a.onclick = async (e) => {
        e.preventDefault(); status(`打开示例 ${a.dataset.n}…`);
        try { onSession(await (await api(`/api/samples/${a.dataset.n}`, { method: "POST" })).json()); }
        catch (err) { status("打开示例失败: " + err.message, "err"); }
      };
    }
  } catch (_) { /* 旧后端没有该接口 */ }
}

// ---------------- 桌面版：版本、更新、数据目录 ----------------
const tauri = window.__TAURI__?.core;
async function initDesktop() {
  if (!tauri) return;
  let info;
  try { info = await tauri.invoke("app_info"); } catch (_) { return; }
  $("#appInfo").hidden = false;
  $("#appVer").textContent = "v" + info.version;
  $("#btnDataDir").title = info.data_dir;
  const btn = $("#btnUpdate");
  let pending = null;
  async function check(quiet) {
    try {
      const u = await tauri.invoke("check_update");
      if (u.available) {
        pending = u; btn.textContent = `更新到 v${u.version}`; btn.classList.add("has");
        btn.title = u.notes || "";
        if (!quiet) status(`有新版本 v${u.version}，点击右下角“更新到 v${u.version}”安装`);
      } else if (!quiet) status(u.message || "已是最新版本");
    } catch (e) { if (!quiet) status("检查更新失败: " + e, "err"); }
  }
  btn.onclick = async (e) => {
    e.preventDefault();
    if (!pending) { btn.textContent = "检查中…"; await check(false); if (!pending) btn.textContent = "检查更新"; return; }
    if (!confirm(`下载并安装 v${pending.version}？\n安装时程序会关闭并自动重启，数据目录里的会话和缓存不受影响。`)) return;
    status("正在下载更新…");
    try { await tauri.invoke("install_update"); } catch (err) { status("更新失败: " + err, "err"); }
  };
  $("#btnBrowse").hidden = false;
  $("#btnBrowse").onclick = async () => {
    try {
      const p = await tauri.invoke("pick_folder");
      if (p) openLocal(p);
    } catch (err) { status("选择目录失败: " + err, "err"); }
  };
  $("#btnDataDir").onclick = async (e) => {
    e.preventDefault();
    try {
      const r = await tauri.invoke("change_data_dir");
      if (r) status("数据目录已切换到 " + r + "，正在重启后端…");
    } catch (err) { status("切换数据目录失败: " + err, "err"); }
  };
  // 桌面窗口里没有浏览器的“新标签页/下载”，改为弹出保存对话框写到本地
  async function saveAs(url, name) {
    const path = await tauri.invoke("pick_save_path", { name });
    if (!path) return;
    status("导出中…");
    try {
      const buf = await (await api(url)).arrayBuffer();
      await tauri.invoke("save_file", new Uint8Array(buf));
      status("已保存 " + path);
    } catch (err) { status("导出失败: " + (err.message || err), "err"); }
  }
  $("#lnkRender").addEventListener("click", (e) => {
    e.preventDefault(); e.stopImmediatePropagation();
    const href = $("#lnkRender").getAttribute("href");
    if (href) saveAs(href, `${S.frames[S.idx].frame_id || S.idx}_${S.cam}.jpg`.replace(/[\\/:*?"<>|]/g, "_"));
  }, true);
  $("#lnkPly").addEventListener("click", (e) => {
    e.preventDefault(); e.stopImmediatePropagation();
    saveAs(`/api/sessions/${S.sid}/stitch.ply?${bevQuery()}`, "stitch.ply");
  }, true);
  window.__TAURI__.event?.listen("update-progress", (ev) => {
    const { done, total } = ev.payload || {};
    status(total ? `下载更新 ${(done / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB` : "下载更新…");
  });
  setTimeout(() => check(true), 3000);
}

// ---------------- 启动 ----------------
loadAdapters();
loadSamples();
initDesktop();
const sid = new URLSearchParams(location.search).get("s");
if (sid) {
  status("恢复上次的会话…");
  $("#progress").textContent = "恢复会话…";
  api(`/api/sessions/${sid}`).then((r) => r.json()).then((s) => onSession(s))
    .catch((e) => { $("#progress").textContent = ""; status("上次的会话无法恢复: " + e.message, "warn"); history.replaceState(null, "", "/"); });
}
