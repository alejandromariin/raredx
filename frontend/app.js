// raredx interface: three views, case -> review -> differential.
// It talks to the API in raredx/api.py, served from the same address.
// No framework: the state lives in `state`, and each view has a render function.

"use strict";

const EXAMPLES = [   // illustrative notes, not real patients
  { note: "3-year-old girl who does not walk yet, with frequent seizures, a small head for her age and low-set ears. Hearing and heart are normal.", age: 3, sex: "female" },
  { note: "17-year-old boy, very tall and thin, with long slender fingers and toes. Both lenses are dislocated upwards. Echocardiogram shows a dilated aortic root. Scoliosis and flat feet. Normal intelligence; no hearing loss.", age: 17, sex: "male" },
  { note: "2-year-old girl who lost the ability to sit and babble after a febrile illness at 14 months. Marked hypotonia, nystagmus and episodes of fast breathing. Lactate is high in blood and cerebrospinal fluid; MRI shows symmetric lesions in the basal ganglia. No seizures.", age: 2, sex: "female" },
];

const state = {
  models: [],            // LLMs used by the server, e.g. ["openai/gpt-6-luna", ...]
  caseId: null,
  finished: false,       // the server case already has a result
  note: "", sex: "", age: null,
  terms: [],             // {hpo, name, present, quote, how, score, alternatives}
  unmatched: [],         // features the server could not code
};

const $ = (id) => document.getElementById(id);
const ICON_X = '<svg class="icon" viewBox="0 0 24 24"><path d="M18 6 6 18M6 6l12 12"/></svg>';
const ICON_EXT = '<svg class="icon ext" viewBox="0 0 24 24"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/></svg>';
const sexValue = () => (document.querySelector('input[name="sex"]:checked') || {}).value || "";
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ------------------------------------------------------------------ helpers
const MODEL_LABELS = { "openai/gpt-6-luna": "GPT-6 Luna", "gemini/gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite" };
function modelLabel(m) {
  if (MODEL_LABELS[m]) return MODEL_LABELS[m];
  return m.split("/").pop().replace(/-/g, " ").replace(/\b(gpt|llm)\b/gi, (x) => x.toUpperCase()).replace(/\b\w/g, (c) => c.toUpperCase());
}

function showError(msg) {
  const el = $("error");
  el.textContent = msg || "";
  el.hidden = !msg;
}

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not JSON */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res;
}

function goTo(view) {
  for (const v of ["case", "review", "result"]) $(`view-${v}`).hidden = v !== view;
  const order = ["case", "review", "result"];
  document.querySelectorAll("#steps li").forEach((li) => {
    const i = order.indexOf(li.dataset.step), cur = order.indexOf(view);
    li.classList.toggle("active", i === cur);
    li.classList.toggle("done", i < cur);
  });
  showError("");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ------------------------------------------------------------------ 1. case
function loadExample(i) {
  const ex = EXAMPLES[i];
  $("note").value = ex.note;
  $("age").value = ex.age;
  document.querySelector(`input[name="sex"][value="${ex.sex}"]`).checked = true;
  $("note").focus();
}

async function extract() {
  const note = $("note").value.trim();
  if (!note) { showError("Write a clinical note first."); return; }
  state.note = note;
  state.sex = sexValue();
  state.age = $("age").value === "" ? null : Number($("age").value);
  const btn = $("btn-extract");
  btn.disabled = true;
  $("extract-status").innerHTML = '<span class="spinner"></span>Reading the note and coding the findings…';
  showError("");
  try {
    const res = await api("/cases", { method: "POST", body: JSON.stringify({ note, sex: state.sex || null, age_years: state.age }) });
    const data = await res.json();
    state.caseId = data.case_id;
    state.finished = false;
    state.terms = [];
    state.unmatched = [];
    for (const f of data.features) {
      if (!f.hpo) { state.unmatched.push(f); continue; }
      addTerm({ hpo: f.hpo, name: f.hpo_name, present: f.present, quote: f.quote, how: f.how, score: f.score, alternatives: f.alternatives || [] });
    }
    renderReview();
    goTo("review");
  } catch (err) {
    showError(`Could not read the note: ${err.message}`);
  } finally {
    btn.disabled = false;
    $("extract-status").textContent = "";
  }
}

// ------------------------------------------------------------------ 2. review
function addTerm(t) {
  const existing = state.terms.find((x) => x.hpo === t.hpo);
  if (existing) { if (t.present) existing.present = true; return; }   // present wins over absent
  state.terms.push(t);
}

function noteWithMarks() {
  // Highlight in the note the words each finding came from.
  const text = state.note, lower = text.toLowerCase();
  const ranges = [];
  state.terms.forEach((t, i) => {
    const q = (t.quote || "").trim();
    if (q.length < 3) return;
    const at = lower.indexOf(q.toLowerCase());
    if (at < 0 || ranges.some((r) => at < r.end && at + q.length > r.start)) return;
    ranges.push({ start: at, end: at + q.length, i, absent: !t.present });
  });
  ranges.sort((a, b) => a.start - b.start);
  let html = "", pos = 0;
  for (const r of ranges) {
    html += esc(text.slice(pos, r.start));
    html += `<mark data-i="${r.i}" class="${r.absent ? "absent" : ""}">${esc(text.slice(r.start, r.end))}</mark>`;
    pos = r.end;
  }
  return html + esc(text.slice(pos));
}

function tagFor(t) {
  if (t.how === "exact") return '<span class="tag">exact match</span>';
  if (t.how === "embedding") return t.score < 0.6 ? '<span class="tag tag-check">check match</span>' : '<span class="tag">closest match</span>';
  if (t.how === "manual") return '<span class="tag">added by you</span>';
  if (t.how === "chosen") return '<span class="tag">chosen by you</span>';
  return "";
}

function termRow(t, i) {
  let from = tagFor(t);
  if (t.quote) from += `<span>from “${esc(t.quote)}”</span>`;
  if (t.how === "embedding" && t.alternatives.length > 1) {
    const opts = t.alternatives.map((a) => `<option value="${esc(a.hpo)}" ${a.hpo === t.hpo ? "selected" : ""}>${esc(a.hpo_name)}</option>`).join("");
    from += `<select data-alt="${i}" aria-label="Choose another term for ${esc(t.name)}">${opts}</select>`;
  }
  return `<li class="term" data-i="${i}">
      <span class="state" role="group" aria-label="Present or absent">
        <button type="button" data-set="present" data-i="${i}" class="${t.present ? "on-present" : ""}" aria-pressed="${t.present}">Present</button>
        <button type="button" data-set="absent" data-i="${i}" class="${t.present ? "" : "on-absent"}" aria-pressed="${!t.present}">Absent</button>
      </span>
      <span class="term-main"><span class="term-name">${esc(t.name)}</span><span class="code">${esc(t.hpo)}</span></span>
      <button type="button" class="icon-btn" data-remove="${i}" title="Remove" aria-label="Remove ${esc(t.name)}">${ICON_X}</button>
      ${from ? `<span class="term-from">${from}</span>` : ""}
    </li>`;
}

function renderReview() {
  $("note-view").innerHTML = state.note ? noteWithMarks() : '<span class="muted">Symptoms entered as codes.</span>';
  const present = [], absent = [];
  state.terms.forEach((t, i) => (t.present ? present : absent).push(termRow(t, i)));
  $("list-present").innerHTML = present.join("") || '<li class="empty">None yet. Add at least one finding below.</li>';
  $("list-absent").innerHTML = absent.join("") || '<li class="empty">None stated in the note.</li>';
  $("count-present").textContent = present.length;
  $("count-absent").textContent = absent.length;

  $("unmatched-block").hidden = state.unmatched.length === 0;
  $("list-unmatched").innerHTML = state.unmatched.map((f, j) => {
    const opts = (f.alternatives || []).map((a) => `<option value="${esc(a.hpo)}">${esc(a.hpo_name)}</option>`).join("");
    return `<li class="term">
        <span class="tag tag-check">not coded</span>
        <span class="term-main"><span class="term-name">${esc(f.term)}</span></span>
        <button type="button" class="icon-btn" data-drop="${j}" title="Ignore" aria-label="Ignore ${esc(f.term)}">${ICON_X}</button>
        <span class="term-from">${f.quote ? `<span>from “${esc(f.quote)}”</span>` : ""}<select data-unmatched="${j}" aria-label="Choose a term"><option value="">Choose a term…</option>${opts}</select></span>
      </li>`;
  }).join("");

  $("btn-run").disabled = present.length === 0;
  const n = state.models.length;
  $("run-hint").textContent = `Ontology ranker${n ? ` + ${n} language model${n > 1 ? "s" : ""}` : ""} in parallel · about 10 s`;
}

function onReviewClick(e) {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.set !== undefined) state.terms[+b.dataset.i].present = b.dataset.set === "present";
  else if (b.dataset.remove !== undefined) state.terms.splice(+b.dataset.remove, 1);
  else if (b.dataset.drop !== undefined) state.unmatched.splice(+b.dataset.drop, 1);
  else return;
  renderReview();
}

function onReviewChange(e) {
  const s = e.target;
  if (s.dataset.alt !== undefined) {               // another alternative for a matched finding
    const t = state.terms[+s.dataset.alt];
    const a = t.alternatives.find((x) => x.hpo === s.value);
    if (a && !state.terms.some((x) => x.hpo === a.hpo && x !== t)) { t.hpo = a.hpo; t.name = a.hpo_name; t.score = a.score; }
  } else if (s.dataset.unmatched !== undefined && s.value) {   // a term chosen for an uncoded finding
    const f = state.unmatched[+s.dataset.unmatched];
    const a = f.alternatives.find((x) => x.hpo === s.value);
    addTerm({ hpo: a.hpo, name: a.hpo_name, present: f.present, quote: f.quote, how: "chosen", score: a.score, alternatives: [] });
    state.unmatched.splice(+s.dataset.unmatched, 1);
  } else return;
  renderReview();
}

function onReviewHover(e) {   // hovering a finding highlights its words in the note
  const li = e.target.closest(".term[data-i]");
  document.querySelectorAll(".note-view mark.hot").forEach((m) => m.classList.remove("hot"));
  if (li) document.querySelector(`.note-view mark[data-i="${li.dataset.i}"]`)?.classList.add("hot");
}

// --- search box to add a finding
let searchTimer = null, searchItems = [], searchSel = -1;
function onSearchInput() {
  clearTimeout(searchTimer);
  const q = $("term-search").value.trim();
  if (q.length < 2) { hideSearch(); return; }
  searchTimer = setTimeout(async () => {
    try {
      const res = await api(`/hpo/search?q=${encodeURIComponent(q)}&limit=8`);
      searchItems = await res.json();
      searchSel = searchItems.length ? 0 : -1;
      renderSearch();
    } catch (err) { hideSearch(); }
  }, 180);
}
function renderSearch() {
  const ul = $("search-results");
  if (!searchItems.length) { ul.innerHTML = '<li class="muted">No matching finding</li>'; ul.hidden = false; return; }
  ul.innerHTML = searchItems.map((r, i) => `<li role="option" data-k="${i}" aria-selected="${i === searchSel}">
      <span>${esc(r.name)}${r.matched.toLowerCase() !== r.name.toLowerCase() ? ` <span class="alt">(${esc(r.matched)})</span>` : ""}</span>
      <span class="term-code">${esc(r.hpo)}</span></li>`).join("");
  ul.hidden = false;
}
function hideSearch() { $("search-results").hidden = true; searchItems = []; searchSel = -1; }
function pickSearch(i) {
  const r = searchItems[i];
  if (!r) return;
  addTerm({ hpo: r.hpo, name: r.name, present: true, quote: "", how: "manual", score: 1, alternatives: [] });
  $("term-search").value = "";
  hideSearch();
  renderReview();
}
function onSearchKey(e) {
  if ($("search-results").hidden) return;
  if (e.key === "ArrowDown") { searchSel = Math.min(searchSel + 1, searchItems.length - 1); renderSearch(); e.preventDefault(); }
  else if (e.key === "ArrowUp") { searchSel = Math.max(searchSel - 1, 0); renderSearch(); e.preventDefault(); }
  else if (e.key === "Enter") { pickSearch(searchSel); e.preventDefault(); }
  else if (e.key === "Escape") hideSearch();
}

// ------------------------------------------------------------------ 3. differential
function progressItems() {
  return [{ step: "ranker", label: "Ontology ranker" }]
    .concat(state.models.map((m) => ({ step: stepOf(m), label: modelLabel(m) })))
    .concat([{ step: "fuse", label: "Learned fusion" }]);
}
const stepOf = (model) => "llm_" + model.split("/").pop().replace(/[^a-z0-9]/gi, "_");

function setStep(step, cls, text) {
  const li = document.querySelector(`#progress li[data-step="${step}"]`);
  if (!li) return;
  li.className = cls;
  if (text !== undefined) li.querySelector(".t").textContent = text;
}

function skeletonRows(n) {
  return Array.from({ length: n }, (_, i) => `<li class="cand skeleton">
      <span class="cand-rank">${i + 1}</span>
      <span class="cand-body"><span class="sk" style="width:${55 + ((i * 17) % 35)}%"></span><span class="sk" style="width:40%;margin-top:10px;height:9px"></span></span>
      <span class="sk"></span></li>`).join("");
}

async function readEvents(res, onEvent) {
  // Server-Sent Events over fetch: blocks separated by a blank line, "event:" and "data:" lines.
  const reader = res.body.getReader(), decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, cut); buf = buf.slice(cut + 2);
      let event = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) onEvent(event, JSON.parse(data));
    }
  }
}

async function runDifferential() {
  const present = state.terms.filter((t) => t.present).map((t) => t.hpo);
  const absent = state.terms.filter((t) => !t.present).map((t) => t.hpo);
  if (!present.length) { showError("Add at least one present finding."); return; }
  $("btn-run").disabled = true;

  const items = progressItems();
  $("progress").innerHTML = items.map((p) => `<li data-step="${p.step}" class="${p.step === "fuse" ? "" : "running"}"><span class="mark"></span>${esc(p.label)}<span class="t">${p.step === "fuse" ? "waiting" : "working…"}</span></li>`).join("");
  $("ranking").innerHTML = skeletonRows(6);
  $("warnings").innerHTML = "";
  $("result-meta").hidden = true;
  $("result-foot").hidden = true;
  goTo("result");
  const started = performance.now();

  try {
    if (state.finished) {   // symptoms changed after a result: start a new server case from the codes
      const res = await api("/cases", { method: "POST", body: JSON.stringify({ hpo: present, hpo_excluded: absent, sex: state.sex || null, age_years: state.age }) });
      state.caseId = (await res.json()).case_id;
    }
    const res = await api(`/cases/${state.caseId}/resume`, { method: "POST", body: JSON.stringify({ hpo_present: present, hpo_excluded: absent }) });
    await readEvents(res, (event, data) => {
      if (event === "step") {
        if (data.step !== "running") setStep(data.step, "done", data.seconds != null ? `${data.seconds} s` : "done");
        // all sources answered (or failed): the fusion is next
        const sourcesLeft = [...document.querySelectorAll("#progress li.running")].filter((li) => li.dataset.step !== "fuse");
        const fuse = document.querySelector('#progress li[data-step="fuse"]');
        if (!sourcesLeft.length && fuse && fuse.className === "") setStep("fuse", "running", "working…");
      } else if (event === "result") {
        state.finished = true;
        renderResult(data, (performance.now() - started) / 1000);
      } else if (event === "error") {
        throw new Error(data.message);
      }
    });
  } catch (err) {
    showError(`The differential could not be computed: ${err.message}`);
  } finally {
    $("btn-run").disabled = false;
  }
}

function renderResult(data, seconds) {
  // bar = relative support among the ten shown (softmax of the fusion scores), not a probability
  const scores = data.ranking.map((r) => r.score);
  const top = Math.max(...scores);
  const w = scores.map((s) => Math.exp(s - top));
  const maxW = Math.max(...w);

  $("ranking").innerHTML = data.ranking.map((r, i) => {
    const omim = r.disease.startsWith("OMIM:") ? r.disease.slice(5) : null;
    const name = omim
      ? `<a href="https://omim.org/entry/${omim}" target="_blank" rel="noopener" title="Open in OMIM">${esc(r.name)}${ICON_EXT}</a>`
      : esc(r.name);
    const sources = [`<span class="src">Ontology <b>#${r.ranker_position}</b></span>`]
      .concat(Object.entries(r.llm_positions).map(([m, pos]) => pos
        ? `<span class="src">${esc(modelLabel(m))} <b>#${pos}</b></span>`
        : `<span class="src off">${esc(modelLabel(m))} —</span>`))
      .join("");
    return `<li class="cand">
        <span class="cand-rank">${r.rank}</span>
        <div class="cand-body">
          <div class="cand-name">${name}</div>
          <div class="cand-meta"><span class="code">${esc(r.disease)}</span>${sources}</div>
        </div>
        <span class="bar" title="Relative support among these ten"><span style="width:${((w[i] / maxW) * 100).toFixed(1)}%"></span></span>
        <div class="cand-details">${renderDetails(r)}</div>
      </li>`;
  }).join("");

  for (const f of data.failed || []) setStep(stepOf(f.split(":")[0]), "failed", "no answer");
  setStep("fuse", "done", undefined);
  $("warnings").innerHTML = (data.failed || []).map((f) => `<p class="alert">${esc(modelLabel(f.split(":")[0]))} did not answer (quota or service overloaded). The ranking uses the other sources.</p>`).join("");
  const usd = data.usd ? ` · $${data.usd.toFixed(4)}` : "";
  $("result-meta").textContent = `${seconds.toFixed(1)} s${usd}`;
  $("result-meta").hidden = false;
  $("result-foot").hidden = false;
}

// Explanations per candidate (why it fits, next test, evidence) will be added here.
function renderDetails(candidate) { return ""; }

// ------------------------------------------------------------------ start
async function init() {
  try { state.models = (await (await api("/health")).json()).llm_models || []; } catch (_) { /* server not ready */ }
  document.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", () => loadExample(+b.dataset.example)));
  $("btn-extract").addEventListener("click", extract);
  $("note").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) extract(); });
  $("view-review").addEventListener("click", onReviewClick);
  $("view-review").addEventListener("change", onReviewChange);
  $("view-review").addEventListener("mouseover", onReviewHover);
  $("term-search").addEventListener("input", onSearchInput);
  $("term-search").addEventListener("keydown", onSearchKey);
  $("search-results").addEventListener("mousedown", (e) => { const li = e.target.closest("li[data-k]"); if (li) pickSearch(+li.dataset.k); });
  $("term-search").addEventListener("blur", () => setTimeout(hideSearch, 150));
  $("btn-back").addEventListener("click", () => goTo("case"));
  $("btn-run").addEventListener("click", runDifferential);
  $("btn-edit").addEventListener("click", () => { renderReview(); goTo("review"); });
  const newCase = () => { $("note").value = ""; $("age").value = ""; $("sex-n").checked = true; state.caseId = null; state.finished = false; state.terms = []; state.unmatched = []; goTo("case"); $("note").focus(); };
  $("btn-new").addEventListener("click", newCase);
  $("btn-new-top").addEventListener("click", newCase);
  goTo("case");
}
init();
