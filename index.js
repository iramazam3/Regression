async function runAllModels() {
  const x = new Float32Array(10);
  for (let i = 0; i < 10; i++) {
    x[i] = parseFloat(document.getElementById(`input${i}`).value) || 0;
  }

  const tensorX = new ort.Tensor("float32", x, [1, 10]);

  await Promise.all([
    // runModel("Linear Regression", "./Linear_Regression.onnx", tensorX, "predLinear"),
    runModel("MLP", "./DLNet_WalmartData.onnx", tensorX, "predMLP"),
    // runModel("Deep Learning with 2 layers", "./Deep_Regression.onnx", tensorX, "predDeep"),
    // runModel("Linear plus Nonlinear", "./Hybrid_Regression.onnx", tensorX, "predHybrid")
  ]);
}


async function runModel(name, modelPath, tensorX, divId) {
  const div = document.getElementById(divId);
  div.innerHTML = `<p>Running ${name}...</p>`;

  try {
    const session = await ort.InferenceSession.create(modelPath + "?v=" + Date.now());
    const results = await session.run({ input1: tensorX });

    // handle the first output tensor
    const output = Object.values(results)[0].data;
    const prediction = output[0].toFixed(2);

    div.innerHTML = `
      <h3>${name}</h3>
      <p><b>Predicted Sales:</b> $${prediction}</p>`;
  } catch (e) {
    console.error(`Error in ${name}:`, e);
    div.innerHTML = `<p style="color:red;">Error: ${e.message}</p>`;
  }
}
