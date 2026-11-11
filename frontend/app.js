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
  $("sex").value = ex.sex;
  $("note").focus();
}

async function extract() {
  const note = $("note").value.trim();
  if (!note) { showError("Write a clinical note first."); return; }
  state.note = note;
  state.sex = $("sex").value;
  state.age = $("age").value === "" ? null : Number($("age").value);
  const btn = $("btn-extract");
  btn.disabled = true;
  $("extract-status").textContent = "Reading the note…";
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

function termRow(t, i) {
  const uncertain = t.how === "embedding" && t.score < 0.6;
  let from = t.quote ? `from “${esc(t.quote)}”` : (t.how === "manual" ? "added by you" : "");
  if (t.how === "embedding" && t.alternatives.length > 1) {
    const opts = t.alternatives.map((a) => `<option value="${esc(a.hpo)}" ${a.hpo === t.hpo ? "selected" : ""}>${esc(a.hpo_name)}</option>`).join("");
    from += `${from ? " · " : ""}${uncertain ? "check this match:" : "closest match:"} <select data-alt="${i}" aria-label="Choose another term">${opts}</select>`;
  }
  return `<li class="term${uncertain ? " uncertain" : ""}" data-i="${i}">
      <span class="term-name">${esc(t.name)}</span>
      <span class="term-code">${esc(t.hpo)}</span>
      <span class="term-actions">
        <button type="button" data-toggle="${i}">${t.present ? "mark absent" : "mark present"}</button>
        <button type="button" data-remove="${i}">remove</button>
      </span>
      ${from ? `<span class="term-from">${from}</span>` : ""}
    </li>`;
}

function renderReview() {
  $("note-view").innerHTML = state.note ? noteWithMarks() : '<span class="muted">Symptoms entered as codes.</span>';
  const present = [], absent = [];
  state.terms.forEach((t, i) => (t.present ? present : absent).push(termRow(t, i)));
  $("list-present").innerHTML = present.join("") || '<li class="empty">None yet. Add at least one finding.</li>';
  $("list-absent").innerHTML = absent.join("") || '<li class="empty">None stated.</li>';
  $("count-present").textContent = present.length;
  $("count-absent").textContent = absent.length;

  $("unmatched-block").hidden = state.unmatched.length === 0;
  $("list-unmatched").innerHTML = state.unmatched.map((f, j) => {
    const opts = (f.alternatives || []).map((a) => `<option value="${esc(a.hpo)}">${esc(a.hpo_name)}</option>`).join("");
    return `<li class="term uncertain">
        <span class="term-name">${esc(f.term)}</span><span></span>
        <span class="term-actions"><button type="button" data-drop="${j}">ignore</button></span>
        <span class="term-from">${f.quote ? `from “${esc(f.quote)}” · ` : ""}choose a term:
          <select data-unmatched="${j}"><option value="">—</option>${opts}</select></span>
      </li>`;
  }).join("");

  $("btn-run").disabled = present.length === 0;
  const n = state.models.length;
  $("run-hint").textContent = `Asks the ontology ranker${n ? ` and ${n} language model${n > 1 ? "s" : ""}` : ""} in parallel; about 10 seconds.`;
}

function onReviewClick(e) {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.toggle !== undefined) { const t = state.terms[+b.dataset.toggle]; t.present = !t.present; }
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
    .concat(state.models.map((m) => ({ step: "llm_" + m.split("/").pop().replace(/[^a-z0-9]/gi, "_"), label: modelLabel(m) })))
    .concat([{ step: "fuse", label: "Learned fusion" }]);
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
  $("progress").innerHTML = items.map((p) => `<li data-step="${p.step}">${esc(p.label)}<span class="t"></span></li>`).join("");
  $("progress").hidden = false;
  $("ranking").innerHTML = "";
  $("warnings").innerHTML = "";
  $("result-meta").textContent = "";
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
        const li = document.querySelector(`#progress li[data-step="${data.step}"]`);
        if (li) { li.classList.add("done"); if (data.seconds != null) li.querySelector(".t").textContent = `${data.seconds} s`; }
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
  const sum = w.reduce((a, b) => a + b, 0) || 1;
  const maxShare = Math.max(...w) / sum;

  $("ranking").innerHTML = data.ranking.map((r, i) => {
    const omim = r.disease.startsWith("OMIM:") ? r.disease.slice(5) : null;
    const name = omim ? `<a href="https://omim.org/entry/${omim}" target="_blank" rel="noopener">${esc(r.name)}</a>` : esc(r.name);
    const sources = [`Ontology <b>#${r.ranker_position}</b>`]
      .concat(Object.entries(r.llm_positions).map(([m, pos]) => `${esc(modelLabel(m))} ${pos ? `<b>#${pos}</b>` : "—"}`))
      .join(" · ");
    const width = ((w[i] / sum) / maxShare) * 100;
    return `<li class="cand">
        <span class="cand-rank">${r.rank}</span>
        <span class="cand-name">${name}<span class="cand-code">${esc(r.disease)}</span></span>
        <span class="bar" title="relative support"><span style="width:${width.toFixed(1)}%"></span></span>
        <span class="cand-sources">${sources}</span>
        <div class="cand-details">${renderDetails(r)}</div>
      </li>`;
  }).join("");

  for (const f of data.failed || []) {   // mark the step that did not answer in the progress list
    const step = "llm_" + f.split(":")[0].split("/").pop().replace(/[^a-z0-9]/gi, "_");
    const li = document.querySelector(`#progress li[data-step="${step}"] .t`);
    if (li) li.textContent = "no answer";
  }
  $("warnings").innerHTML = (data.failed || []).map((f) => {
    const model = f.split(":")[0];
    return `<p class="warning">${esc(modelLabel(model))} did not answer (quota or service overloaded). The ranking uses the other sources.</p>`;
  }).join("");
  const usd = data.usd ? ` · $${data.usd.toFixed(4)}` : "";
  $("result-meta").textContent = `${seconds.toFixed(1)} s${usd}`;
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
  $("btn-new").addEventListener("click", () => { $("note").value = ""; state.caseId = null; state.terms = []; state.unmatched = []; goTo("case"); });
  goTo("case");
}
init();
