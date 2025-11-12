// Collect inputs as Float32Array
function collectInputs() {
  const x = new Float32Array(17);
  for (let i = 0; i < 17; i++) {
    x[i] = parseFloat(document.getElementById(`input${i}`).value) || 0;
  }
  return x;
}

// Format numbers with commas
function formatCurrency(num) {
  return num.toLocaleString("en-US", { style: "currency", currency: "USD" });
}

// Run ONNX Model
async function runSelectedModel() {
  const modelFile = document.getElementById("modelSelect").value;
  const outputText = document.getElementById("outputText");
  const btn = document.getElementById("runBtn");

  try {
    btn.disabled = true;
    outputText.textContent = `Loading ${modelFile}...`;

    const x = collectInputs();
    const tensorX = new ort.Tensor("float32", x, [1, 17]);

    const session = await ort.InferenceSession.create(`./${modelFile}?v=${Date.now()}`);
    const inputName = session.inputNames[0] || "input";
    const results = await session.run({ [inputName]: tensorX });

    const outputTensor = results[session.outputNames?.[0]] || Object.values(results)[0];
    const prediction = outputTensor.data[0];

    outputText.innerHTML = `
      <div><b>Model:</b> ${modelFile.replace(".onnx","")}</div>
      <div><b>Predicted Weekly Sales:</b> ${formatCurrency(prediction)}</div>
    `;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    outputText.innerHTML = `<span style="color:red;">Error: ${e.message}</span>`;
  } finally {
    validateInputs();
  }
}

// Button listener
window.addEventListener("load", () => {
  document.getElementById("runBtn").addEventListener("click", runSelectedModel);
});
