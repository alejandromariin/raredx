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
  summaryModel: null,    // LLM that writes the one-sentence summaries
  ranking: [],           // current differential
  explain: null,         // {explanations, next_findings} for the top candidates
  summaries: {},         // {disease: sentence}
  summariesDone: false,  // the summary step finished (with or without sentences)
  evidenceOn: false,     // the server can search the web (Tavily key set)
  evidence: {},          // {disease: [live web results]}
  evidenceDone: false,
  caseId: null,
  finished: false,       // the server case already has a result
  note: "", sex: "", age: null,
  terms: [],             // {hpo, name, present, quote, how, score, alternatives}
  unmatched: [],         // features the server could not code
};

const $ = (id) => document.getElementById(id);
const ICON_X = '<svg class="icon" viewBox="0 0 24 24"><path d="M18 6 6 18M6 6l12 12"/></svg>';
const ICON_DOWN = '<svg class="icon" viewBox="0 0 24 24"><path d="m6 9 6 6 6-6"/></svg>';
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
    .concat([{ step: "fuse", label: "Learned fusion" },
             { step: "explain", label: "Explanations", via: "HPO annotations" },
             { step: "summarize", label: "Summary", via: state.summaryModel ? modelLabel(state.summaryModel) : "LLM" }])
    .concat(state.evidenceOn ? [{ step: "evidence", label: "Live evidence", via: "Tavily web search" }] : []);
}
// After the sources: fuse, then explain, then summary and live evidence at the same time
const STAGES = [["fuse"], ["explain"], ["summarize", "evidence"]];
const AFTER_SOURCES = STAGES.flat();
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
  $("progress").innerHTML = items.map((p) => {
    const later = AFTER_SOURCES.includes(p.step);
    return `<li data-step="${p.step}" class="${later ? "" : "running"}"><span class="mark"></span>${esc(p.label)}${p.via ? `<span class="via">${esc(p.via)}</span>` : ""}<span class="t">${later ? "waiting" : "working…"}</span></li>`;
  }).join("");
  $("next-card").innerHTML = "";
  state.ranking = []; state.explain = null; state.summaries = {}; state.summariesDone = false;
  state.evidence = {}; state.evidenceDone = false;
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
        // when the sources are done, the steps after them start one after the other
        const sourcesLeft = [...document.querySelectorAll("#progress li.running")].filter((li) => !AFTER_SOURCES.includes(li.dataset.step));
        if (!sourcesLeft.length) {
          const li = (st) => document.querySelector(`#progress li[data-step="${st}"]`);
          for (const stage of STAGES) {          // start the first stage that has not finished
            const present = stage.filter((st) => li(st));
            if (present.every((st) => ["done", "failed"].includes(li(st).className))) continue;
            present.filter((st) => li(st).className === "").forEach((st) => setStep(st, "running", "working…"));
            break;
          }
        }
      } else if (event === "ranking") {
        renderRanking(data.ranking);
      } else if (event === "explain") {
        state.explain = data;
        renderExplanations();
      } else if (event === "evidence") {
        state.evidence = data.evidence || {};
        state.evidenceDone = true;
        renderSummaries();
      } else if (event === "summaries") {
        state.summaries = data.summaries || {};
        state.summariesDone = true;
        renderSummaries();
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

function renderRanking(ranking) {
  // bar = relative support among the ten shown (softmax of the fusion scores), not a probability
  state.ranking = ranking;
  const scores = ranking.map((r) => r.score);
  const top = Math.max(...scores);
  const w = scores.map((s) => Math.exp(s - top));
  const maxW = Math.max(...w);

  $("ranking").innerHTML = ranking.map((r, i) => {
    const omim = r.disease.startsWith("OMIM:") ? r.disease.slice(5) : null;
    const name = omim
      ? `<a href="https://omim.org/entry/${omim}" target="_blank" rel="noopener" title="Open in OMIM">${esc(r.name)}${ICON_EXT}</a>`
      : esc(r.name);
    const sources = [`<span class="src">Ontology <b>#${r.ranker_position}</b></span>`]
      .concat(Object.entries(r.llm_positions).map(([m, pos]) => pos
        ? `<span class="src">${esc(modelLabel(m))} <b>#${pos}</b></span>`
        : `<span class="src off">${esc(modelLabel(m))} —</span>`))
      .join("");
    return `<li class="cand" data-d="${esc(r.disease)}">
        <span class="cand-rank">${r.rank}</span>
        <div class="cand-body">
          <div class="cand-name">${name}</div>
          <div class="cand-meta"><span class="code">${esc(r.disease)}</span>${sources}<span class="why-slot"></span></div>
        </div>
        <span class="bar" title="Relative support among these ten"><span style="width:${((w[i] / maxW) * 100).toFixed(1)}%"></span></span>
        <div class="cand-details" hidden></div>
      </li>`;
  }).join("");
  if (state.explain) renderExplanations();
}

function renderResult(data, seconds) {
  if (!state.ranking.length) renderRanking(data.ranking);
  if (!state.explain && data.explanations && Object.keys(data.explanations).length) {
    state.explain = { explanations: data.explanations, next_findings: data.next_findings || [] };
    renderExplanations();
  }
  state.summaries = data.summaries || state.summaries;
  state.summariesDone = true;
  state.evidence = data.evidence || state.evidence;
  state.evidenceDone = true;
  renderSummaries();
  for (const f of data.failed || []) {
    if (f.includes("(summary)")) setStep("summarize", "failed", "no answer");
    else if (f.includes("(evidence)")) setStep("evidence", "failed", "no answer");
    else setStep(stepOf(f.split(":")[0]), "failed", "no answer");
  }
  for (const st of AFTER_SOURCES) {
    const li = document.querySelector(`#progress li[data-step="${st}"]`);
    if (li && li.className !== "failed") setStep(st, "done", li.className === "done" ? undefined : "done");
  }
  $("warnings").innerHTML = (data.failed || []).filter((f) => !f.includes("(summary)") && !f.includes("(evidence)")).map((f) =>
    `<p class="alert">${esc(modelLabel(f.split(":")[0]))} did not answer (quota or service overloaded). The ranking uses the other sources.</p>`).join("");
  const usd = data.usd ? ` · $${data.usd.toFixed(4)}` : "";
  $("result-meta").textContent = `${seconds.toFixed(1)} s${usd}`;
  $("result-meta").hidden = false;
  $("result-foot").hidden = false;
}

// ------------------------------------------------------------------ explanations
const shortName = (name) => name.split(/[,;]/)[0];
const pct = (f) => `${Math.round(f * 100)}%`;
const chipList = (items, cls, extra) => items.length
  ? items.map((x) => `<span class="fchip ${cls}" title="${esc(x.hpo)}">${esc(x.name)}${extra ? extra(x) : ""}</span>`).join("")
  : '<span class="fchip none">none</span>';

function whyPanel(disease) {
  const e = state.explain.explanations[disease];
  const summary = state.summaries[disease];
  const summaryModel = state.summaryModel ? modelLabel(state.summaryModel) : "LLM";
  const fits = e.fits.map((x) => ({ ...x, name: x.as ? `${x.name} ~ ${x.as}` : x.name })).concat(e.partial.map((x) => ({ ...x, name: `${x.name} ~ ${x.shared}` })));
  return `<div class="why">
      <div class="why-summary">${summary
        ? `<p class="why-text">${esc(summary)}</p><span class="via">Written by ${esc(summaryModel)} from the facts below</span>`
        : state.summariesDone
          ? `<span class="via">No summary available for this candidate; the facts below come from the HPO files.</span>`
          : `<span class="sk-line" style="width:92%"></span><span class="sk-line" style="width:64%"></span><span class="via">${esc(summaryModel)} is writing a summary…</span>`}</div>
      <div class="why-grid">
        <div class="why-block"><h4>Fits</h4><div class="fchips">${chipList(fits, "fit")}</div></div>
        <div class="why-block"><h4>Against</h4><div class="fchips">${chipList(e.against, "against", (x) => ` <small>absent · seen in ${pct(x.frequency)}</small>`)}</div></div>
        <div class="why-block"><h4>Typical, not mentioned</h4><div class="fchips">${chipList(e.typical, "typ", (x) => ` <small>${pct(x.frequency)}</small>`)}</div></div>
        <div class="why-block"><h4>Not explained by it</h4><div class="fchips">${chipList(e.unexplained, "typ")}</div></div>
      </div>
      <div class="why-foot">${e.genes.length
        ? `<span>Confirm with genetic testing:</span>${e.genes.map((g) => `<span class="gene">${esc(g)}</span>`).join("")}`
        : '<span class="muted">No gene linked in the HPO files.</span>'}
        <span class="via">Findings and frequencies: HPO annotations (phenotype.hpoa) · Genes: HPO genes_to_disease</span></div>
      ${evidenceBlock(disease)}
    </div>`;
}

const EVIDENCE_GROUPS = [["literature", "Recent literature"], ["trials", "Clinical trials"], ["support", "Patient information and support"]];
function evidenceBlock(disease) {
  if (!state.evidenceOn) return "";
  const rank = state.ranking.findIndex((r) => r.disease === disease);
  if (!state.evidenceDone) {
    return `<div class="evidence"><h4>Live evidence</h4><span class="sk-line" style="width:80%"></span><span class="sk-line" style="width:55%"></span><span class="via">Tavily is searching the web…</span></div>`;
  }
  const found = state.evidence[disease];
  if (!found) {
    return rank >= 3 ? `<div class="evidence"><h4>Live evidence</h4><span class="via">Live evidence is searched for the top 3 candidates.</span></div>` : "";
  }
  const groups = EVIDENCE_GROUPS.map(([kind, title]) => {
    const items = found.filter((r) => r.kind === kind);
    return `<div class="ev-group ev-${kind}"><h5>${title}</h5>${items.length ? `<ul>${items.map((r) => `<li>
          <a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title)}${ICON_EXT}</a>
          <span class="ev-site">${esc(r.site)}${r.date ? ` · ${esc(String(r.date).slice(0, 10))}` : ""}</span>
          ${r.snippet ? `<p>${esc(r.snippet)}</p>` : ""}</li>`).join("")}</ul>` : '<span class="fchip none">nothing found</span>'}</div>`;
  }).join("");
  return `<div class="evidence"><h4>Live evidence</h4><div class="ev-grid">${groups}</div>
      <span class="via">Found now by Tavily web search (PubMed, GeneReviews, ClinicalTrials.gov, Orphanet, NORD, GARD). Not used for the ranking.</span></div>`;
}

function renderExplanations() {
  const ex = state.explain;
  if (!ex) return;
  document.querySelectorAll("#ranking .cand").forEach((li) => {
    const d = li.dataset.d;
    if (!ex.explanations[d]) return;
    const slot = li.querySelector(".why-slot");
    if (!slot.innerHTML) slot.innerHTML = `<button type="button" class="why-btn" data-why="${esc(d)}" aria-expanded="false">Why ${ICON_DOWN}</button>`;
    const box = li.querySelector(".cand-details");
    if (!box.hidden) box.innerHTML = whyPanel(d);
  });
  // open the first candidate by default
  const first = document.querySelector("#ranking .cand");
  if (first && ex.explanations[first.dataset.d] && first.querySelector(".cand-details").hidden) toggleWhy(first.dataset.d);
  renderNextFindings();
}

function renderSummaries() {
  document.querySelectorAll("#ranking .cand").forEach((li) => {
    const box = li.querySelector(".cand-details");
    if (!box.hidden && state.explain && state.explain.explanations[li.dataset.d]) box.innerHTML = whyPanel(li.dataset.d);
  });
}

function toggleWhy(disease) {
  const li = document.querySelector(`#ranking .cand[data-d="${CSS.escape(disease)}"]`);
  if (!li) return;
  const box = li.querySelector(".cand-details");
  const open = box.hidden;
  if (open) box.innerHTML = whyPanel(disease);
  box.hidden = !open;
  li.classList.toggle("open", open);
  li.querySelector(".why-btn")?.setAttribute("aria-expanded", String(open));
}

function renderNextFindings() {
  const list = (state.explain && state.explain.next_findings) || [];
  if (!list.length) { $("next-card").innerHTML = ""; return; }
  const names = Object.fromEntries(state.ranking.map((r) => [r.disease, shortName(r.name)]));
  $("next-card").innerHTML = `<div class="card next-card">
      <div class="card-head"><span>Most useful next findings</span><span class="via">information gain over the top 5 · HPO annotation frequencies</span></div>
      <p class="next-hint">Checking these in the patient would best tell the top candidates apart. Mark what you find and the differential is recomputed.</p>
      <ul class="next-list">${list.map((f, i) => {
        const yes = Object.entries(f.expected).filter(([, q]) => q >= 0.3).map(([d, q]) => `${esc(names[d] || d)} ${pct(q)}`);
        return `<li>
            <span class="next-name">${esc(f.name)} <span class="code">${esc(f.hpo)}</span></span>
            <span class="next-actions"><button type="button" data-next="${i}" data-val="1">Present</button><button type="button" class="neg" data-next="${i}" data-val="0">Absent</button></span>
            <span class="next-exp">Expected in ${yes.join(" · ") || "none of them"}${yes.length < Object.keys(f.expected).length ? " · rare in the rest" : ""}</span>
          </li>`;
      }).join("")}</ul>
    </div>`;
}

function onResultClick(e) {
  const why = e.target.closest("[data-why]");
  if (why) { toggleWhy(why.dataset.why); return; }
  const nb = e.target.closest("[data-next]");
  if (nb) {   // the doctor answers a suggested finding: add it and recompute
    const f = state.explain.next_findings[+nb.dataset.next];
    addTerm({ hpo: f.hpo, name: f.name, present: nb.dataset.val === "1", quote: "", how: "manual", score: 1, alternatives: [] });
    runDifferential();
  }
}


// ------------------------------------------------------------------ start
async function init() {
  try {
    const h = await (await api("/health")).json();
    state.models = h.llm_models || [];
    state.summaryModel = h.summary_model || null;
    state.evidenceOn = !!h.evidence;
  } catch (_) { /* server not ready */ }
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
  $("view-result").addEventListener("click", onResultClick);
  $("btn-edit").addEventListener("click", () => { renderReview(); goTo("review"); });
  const newCase = () => { $("note").value = ""; $("age").value = ""; $("sex-n").checked = true; state.caseId = null; state.finished = false; state.terms = []; state.unmatched = []; goTo("case"); $("note").focus(); };
  $("btn-new").addEventListener("click", newCase);
  $("btn-new-top").addEventListener("click", newCase);
  goTo("case");
}
init();
