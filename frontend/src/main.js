import "./style.css";
import { createScene } from "./scene.js";
import { buildNeurons, buildConnections } from "./network.js";
import { fetchSamples, fetchInference, fetchWeights } from "./api.js";
import { animateFeedforward } from "./activations.js";

let layerMeshes = null;
let connectionMeshes = null;
let selectedIndex = null;

// Initialize Three.js scene
const container = document.getElementById("canvas-container");
const { scene } = createScene(container);

// Build neuron geometry immediately
layerMeshes = buildNeurons(scene);

// Load weights and build connections
fetchWeights().then((data) => {
  connectionMeshes = buildConnections(scene, data);
});

// --- Digit picker ---
const digitButtonsContainer = document.getElementById("digit-buttons");
for (let d = 0; d < 10; d++) {
  const btn = document.createElement("button");
  btn.textContent = d;
  btn.addEventListener("click", () => selectDigit(d));
  digitButtonsContainer.appendChild(btn);
}

async function selectDigit(digit) {
  // Update active button
  digitButtonsContainer.querySelectorAll("button").forEach((b, i) => {
    b.classList.toggle("active", i === digit);
  });

  // Fetch samples and render thumbnails
  const samples = await fetchSamples(digit);
  renderThumbnails(samples);
  selectedIndex = null;
  document.getElementById("run-btn").disabled = true;
  document.getElementById("result").classList.add("hidden");
}

// --- Sample thumbnails ---
function renderThumbnails(samples) {
  const strip = document.getElementById("sample-thumbnails");
  strip.innerHTML = "";

  for (const sample of samples) {
    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(28, 28);

    for (let i = 0; i < 784; i++) {
      const v = Math.round(sample.pixels[i] * 255);
      imgData.data[i * 4] = v;
      imgData.data[i * 4 + 1] = v;
      imgData.data[i * 4 + 2] = v;
      imgData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);

    canvas.addEventListener("click", () => {
      strip.querySelectorAll("canvas").forEach((c) => c.classList.remove("selected"));
      canvas.classList.add("selected");
      selectedIndex = sample.index;
      document.getElementById("run-btn").disabled = false;
    });

    strip.appendChild(canvas);
  }
}

// --- Run inference ---
document.getElementById("run-btn").addEventListener("click", async () => {
  if (selectedIndex === null || !connectionMeshes) return;

  const btn = document.getElementById("run-btn");
  btn.disabled = true;
  btn.textContent = "Running...";
  document.getElementById("result").classList.add("hidden");

  const data = await fetchInference(selectedIndex);
  await animateFeedforward(layerMeshes, connectionMeshes, data.activations);

  // Show result
  const maxProb = Math.max(...data.activations.probabilities);
  const pct = (maxProb * 100).toFixed(1);
  document.getElementById("result-text").textContent =
    `Predicted: ${data.prediction} (${pct}%)`;
  document.getElementById("result").classList.remove("hidden");

  btn.textContent = "Run Inference";
  btn.disabled = false;
});
