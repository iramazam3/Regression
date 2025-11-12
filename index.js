
// Collect inputs as Float32Array
function collectInputs() {
  const x = new Float32Array(10);
  for (let i = 0; i < 10; i++) {
    x[i] = parseFloat(document.getElementById(`input${i}`).value) || 0;
  }
  return x;
}

// Run selected model
async function runSelectedModel() {
  const modelFile = document.getElementById("modelSelect").value;
  const outputText = document.getElementById("outputText");
  const btn = document.getElementById("runBtn");

  try {
    btn.disabled = true;
    outputText.textContent = `Loading ${modelFile}...`;

    const x = collectInputs();
    const tensorX = new ort.Tensor("float32", x, [1, 10]);

    const session = await ort.InferenceSession.create(`./${modelFile}?v=${Date.now()}`);
    const inputName = session.inputNames[0] || "input";
    const results = await session.run({ [inputName]: tensorX });

    const firstOutput = results[session.outputNames?.[0]] || Object.values(results)[0];
    const output = firstOutput.data;

    // Get predicted class
    const predictedSales = Array.isArray(output) ? output[0] : output;

    // Display result
    outputText.innerHTML = `
      <div><b>Model:</b> ${modelFile.replace(".onnx","")}</div>
      <div><b>Predicted Sales:</b> $${Number(predictedSales).toFixed(2)}</div>
    `;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    outputText.innerHTML = `<span style="color:red;">Error: ${e.message}</span>`;
  } finally {
    validateInputs();
  }
}

// Attach to button
window.addEventListener("load", () => {
  document.getElementById("runBtn").addEventListener("click", runSelectedModel);
});
