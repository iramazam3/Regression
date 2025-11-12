// Format numbers with commas
function formatCurrency(num) {
  return num.toLocaleString("en-US", { style: "currency", currency: "USD" });
}

// Collect inputs
function collectInputs() {
  const x = [];
  for (let i = 0; i < 17; i++) {
    x.push(parseFloat(document.getElementById(`input${i}`).value) || 0);
  }
  return x;
}

// ---------- XGBOOST JS MODEL ----------
function runXGBModel() {
  const input = collectInputs();
  const prediction = score(input);   // score() comes from xgb_model.js

  document.getElementById("outputText").innerHTML = `
    <div><b>Model:</b> XGBoost (JavaScript)</div>
    <div><b>Predicted Weekly Sales:</b> ${formatCurrency(prediction)}</div>
  `;
}

// ---------- ONNX MODEL ----------
async function runONNXModel(modelFile) {
  const outputText = document.getElementById("outputText");

  outputText.textContent = `Loading ${modelFile}...`;

  const x = collectInputs();
  const tensorX = new ort.Tensor("float32", Float32Array.from(x), [1, 17]);

  const session = await ort.InferenceSession.create(`./${modelFile}?v=${Date.now()}`);
  const inputName = session.inputNames[0];
  const results = await session.run({ [inputName]: tensorX });

  const outputTensor = results[session.outputNames[0]];
  const prediction = outputTensor.data[0];

  outputText.innerHTML = `
    <div><b>Model:</b> ${modelFile.replace(".onnx","")}</div>
    <div><b>Predicted Weekly Sales:</b> ${formatCurrency(prediction)}</div>
  `;
}

// ---------- MODEL ROUTER ----------
async function runSelectedModel() {
  const model = document.getElementById("modelSelect").value;

  if (model === "xgboost_js") {
    runXGBModel();
  } else {
    await runONNXModel(model);
  }
}

// Run button listener
window.addEventListener("load", () => {
  document.getElementById("runBtn").addEventListener("click", runSelectedModel);
});
