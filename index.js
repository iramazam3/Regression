// Validate inputs: enable button only when all 17 are filled
function validateInputs() {
  let ok = true;
  for (let i = 0; i < 17; i++) {
    const el = document.getElementById(`input${i}`);
    if (!el || el.value.trim() === "") { ok = false; break; }
  }
  document.getElementById("runBtn").disabled = !ok;
}

// Collect inputs as Float32Array
function collectInputs() {
  const x = new Float32Array(17);
  for (let i = 0; i < 17; i++) {
    x[i] = parseFloat(document.getElementById(`input${i}`).value) || 0;
  }
  return x;
}

// Run selected model
async function runSelectedModel() {
  const outputText = document.getElementById("outputText");
  const btn = document.getElementById("runBtn");
  const modelFile = document.getElementById("modelSelect").value;

  try {
    btn.disabled = true;
    outputText.textContent = `Loading ${modelFile} and running inference...`;

    const x = collectInputs();
    const tensorX = new ort.Tensor("float32", x, [1, 17]);

    // Load session 
    const session = await ort.InferenceSession.create(`./${modelFile}?v=${Date.now()}`);

    // Use the first declared input name from the model
    const inputName = session.inputNames && session.inputNames.length ? session.inputNames[0] : "input1";

    // Run inference
    const results = await session.run({ [inputName]: tensorX });

    // Read first output tensor
    const firstOutput = results[session.outputNames?.[0]] || Object.values(results)[0];
    const data = firstOutput.data;
    const pred = Array.isArray(data) ? data[0] : data[0]; // Float32Array

    // Render prediction
    outputText.innerHTML = `
      <div><b>Model:</b> ${modelFile.replace(".onnx","")}</div>
      <div><b>Predicted Sales:</b> $${Number(pred).toFixed(2)}</div>
    `;
  } catch (e) {
    console.error(e);
    outputText.innerHTML = `<span style="color:#c53030">Error: ${e.message}</span>`;
  } finally {
    // Re-validate to restore proper state
    validateInputs();
  }
}

// Hook up events after page loads
window.addEventListener("load", () => {
  for (let i = 0; i < 17; i++) {
    const el = document.getElementById(`input${i}`);
    el.addEventListener("input", validateInputs);
    el.addEventListener("change", validateInputs);
  }
  document.getElementById("runBtn").addEventListener("click", runSelectedModel);
  validateInputs();
});
