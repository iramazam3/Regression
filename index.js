// --- Validate inputs ---
function validateInputs() {
  let ok = true;
  for (let i = 0; i < 17; i++) {
    const el = document.getElementById(`input${i}`);
    if (!el || el.value.trim() === "") { ok = false; break; }
  }
  document.getElementById("runBtn").disabled = !ok;
}

// --- Collect inputs as Float32Array ---
function collectInputs() {
  const x = new Float32Array(17);
  for (let i = 0; i < 17; i++) {
    x[i] = parseFloat(document.getElementById(`input${i}`).value) || 0;
  }
  return x;
}

// --- Run selected model ---
async function runSelectedModel() {
  const outputText = document.getElementById("outputText");
  const btn = document.getElementById("runBtn");
  const modelFile = document.getElementById("modelSelect").value;

  try {
    btn.disabled = true;
    outputText.textContent = `Running ${modelFile}...`;

    const x = collectInputs();

    // --- JS model case (m2cgen exported) ---
    if (modelFile === "XGBoost_WalmartData") {
      const pred = score(Array.from(x)); // "score" defined in xgb_model.js
      outputText.innerHTML = `
        <div><b>Model:</b> XGBoost</div>
        <div><b>Predicted Sales:</b> $${Number(pred).toFixed(2)}</div>
      `;
      return;
    }

    // --- ONNX model case ---
    const tensorX = new ort.Tensor("float32", x, [1, 17]);
    const session = await ort.InferenceSession.create(`./${modelFile}?v=${Date.now()}`);
    const inputName = session.inputNames && session.inputNames.length ? session.inputNames[0] : "input1";
    const results = await session.run({ [inputName]: tensorX });
    const firstOutput = results[session.outputNames?.[0]] || Object.values(results)[0];
    const data = firstOutput.data;
    const pred = Array.isArray(data) ? data[0] : data[0];

    outputText.innerHTML = `
      <div><b>Model:</b> ${modelFile.replace(".onnx","")}</div>
      <div><b>Predicted Sales:</b> $${Number(pred).toFixed(2)}</div>
    `;
  } catch (e) {
    console.error(e);
    outputText.innerHTML = `<span style="color:#c53030">Error: ${e.message}</span>`;
  } finally {
    validateInputs();
  }
}

// --- Initialize events ---
window.addEventListener("load", () => {
  for (let i = 0; i < 17; i++) {
    const el = document.getElementById(`input${i}`);
    el.addEventListener("input", validateInputs);
    el.addEventListener("change", validateInputs);
  }
  document.getElementById("runBtn").addEventListener("click", runSelectedModel);
  validateInputs();
});
