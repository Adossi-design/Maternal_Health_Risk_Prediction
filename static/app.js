// Maternal Health Risk frontend.
// Builds the form from /api/v1/meta, posts to /api/v1/predict, renders results.

const API = "/api/v1";

const els = {
  fields: document.getElementById("fields"),
  form: document.getElementById("predict-form"),
  result: document.getElementById("result"),
  error: document.getElementById("error"),
  banner: document.getElementById("risk-banner"),
  riskValue: document.getElementById("risk-value"),
  riskConf: document.getElementById("risk-confidence"),
  probBars: document.getElementById("prob-bars"),
  observations: document.getElementById("observations"),
  obsList: document.getElementById("obs-list"),
  obsTitle: null, // set after DOM ready
  submit: document.getElementById("submit-btn"),
};
els.obsTitle = document.querySelector(".obs-title");

let meta = null;

// Map a class label ("low/mid/high risk") to a CSS keyword.
function riskClass(label) {
  if (label.includes("low")) return "low";
  if (label.includes("mid")) return "mid";
  if (label.includes("high")) return "high";
  return "";
}

function showError(msg) {
  els.error.textContent = msg;
  els.error.classList.remove("hidden");
  els.result.classList.add("hidden");
}

async function loadMeta() {
  const res = await fetch(`${API}/meta`);
  if (!res.ok) throw new Error(`meta request failed (${res.status})`);
  meta = await res.json();

  els.fields.innerHTML = "";
  for (const spec of meta.feature_meta) {
    const wrap = document.createElement("div");
    wrap.className = "field";
    wrap.innerHTML = `
      <label for="${spec.name}">${spec.label} <span class="unit">(${spec.unit})</span></label>
      <input id="${spec.name}" name="${spec.name}" type="number"
             min="${spec.min}" max="${spec.max}" placeholder="e.g. ${spec.default}"
             step="${spec.integer ? "1" : "0.1"}" required />`;
    els.fields.appendChild(wrap);
  }
}

function renderResult(data) {
  els.error.classList.add("hidden");
  const rc = riskClass(data.risk_level);

  els.banner.className = `risk-banner ${rc}`;
  els.riskValue.textContent = data.risk_level;
  els.riskConf.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

  const order = ["low risk", "mid risk", "high risk"];
  const classes = order.filter((k) => k in data.probabilities);

  els.probBars.innerHTML = "";
  for (const cls of classes) {
    const p = data.probabilities[cls];
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="name">${cls}</span>
      <span class="prob-track"><span class="prob-fill ${riskClass(cls)}" style="width:0%"></span></span>
      <span class="pct">${(p * 100).toFixed(0)}%</span>`;
    els.probBars.appendChild(row);
    requestAnimationFrame(() => {
      row.querySelector(".prob-fill").style.width = `${(p * 100).toFixed(1)}%`;
    });
  }
  els.result.classList.remove("hidden");
}

// Flag entered values that fall outside their healthy reference range.
function renderObservations(payload) {
  const flagged = [];
  for (const spec of meta.feature_meta) {
    const v = payload[spec.name];
    if (Number.isNaN(v)) continue;
    if (spec.normal_low != null && v < spec.normal_low) {
      flagged.push({ spec, value: v, status: "below" });
    } else if (spec.normal_high != null && v > spec.normal_high) {
      flagged.push({ spec, value: v, status: "above" });
    }
  }

  els.obsList.innerHTML = "";
  if (flagged.length === 0) {
    els.obsTitle.textContent = "All readings are within the normal range";
    els.observations.classList.add("all-clear");
    return;
  }

  els.observations.classList.remove("all-clear");
  els.obsTitle.textContent = "Readings outside the normal range";
  for (const { spec, value, status } of flagged) {
    const li = document.createElement("li");
    li.className = "obs-item";
    const word = status === "above" ? "High" : "Low";
    const normal = `${spec.normal_low}–${spec.normal_high} ${spec.unit}`;
    li.innerHTML = `
      <span class="obs-name">${spec.label}</span>
      <span class="obs-value">${value} ${spec.unit}</span>
      <span class="obs-normal">normal ${normal}</span>
      <span class="obs-tag ${status}">${word}</span>`;
    els.obsList.appendChild(li);
  }
}

const BTN_DEFAULT = els.submit.innerHTML;

els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {};
  for (const spec of meta.feature_meta) {
    payload[spec.name] = parseFloat(document.getElementById(spec.name).value);
  }

  els.submit.disabled = true;
  els.submit.textContent = "Predicting…";
  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed (${res.status})`);
    }
    renderResult(await res.json());
    renderObservations(payload);
  } catch (err) {
    showError(err.message || "Something went wrong.");
  } finally {
    els.submit.disabled = false;
    els.submit.innerHTML = BTN_DEFAULT;
  }
});

loadMeta().catch(() => showError("Could not load model metadata. Is the server running?"));
