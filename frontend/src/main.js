import "./style.css";
import { createScene } from "./scene.js";
import { buildNeurons, buildConnections } from "./network.js";
import { fetchArchitecture, fetchSamples, fetchInference, fetchWeights, setModel } from "./api.js";
import { animateFeedforward } from "./activations.js";
import { buildLayerConfig } from "./constants.js";

let layerMeshes = null;
let connectionMeshes = null;
let selectedIndex = null;
let layers = null;
let weightLayers = null;

// Initialize Three.js scene
const container = document.getElementById("canvas-container");
const { scene, camera, controls } = createScene(container);

// Track all scene objects for teardown
let sceneObjects = [];

function teardownScene() {
  for (const obj of sceneObjects) {
    scene.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) obj.material.dispose();
  }
  sceneObjects = [];
  layerMeshes = null;
  connectionMeshes = null;

  // Clear legend
  document.getElementById("legend-layers").innerHTML = "";
}

// Fetch architecture and weights, then build geometry
async function init() {
  const [archData, weightsData] = await Promise.all([
    fetchArchitecture(),
    fetchWeights(),
  ]);

  const config = buildLayerConfig(archData);
  layers = config.layers;
  weightLayers = config.weightLayers;

  layerMeshes = buildNeurons(scene, layers);
  connectionMeshes = buildConnections(scene, weightsData, layers, weightLayers);

  // Track all added meshes for teardown
  for (const name in layerMeshes) {
    sceneObjects.push(layerMeshes[name]);
  }
  for (const key in connectionMeshes) {
    sceneObjects.push(connectionMeshes[key]);
  }

  // Reposition camera target to center of network z-extent
  const zSpacing = 35;
  const zCenter = (layers.length - 1) * zSpacing / 2;
  controls.target.set(0, 0, zCenter);
  camera.position.set(-70, 25, zCenter + 65);
  controls.update();

  // Build legend dynamically
  const legendContainer = document.getElementById("legend-layers");
  for (const layer of layers) {
    const item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML = `<span class="legend-dot"></span>${layer.displayName}`;
    legendContainer.appendChild(item);
  }

  // Build digit buttons from output layer size
  const outputLayer = layers[layers.length - 1];
  const numDigits = outputLayer.size;
  const digitButtonsContainer = document.getElementById("digit-buttons");
  digitButtonsContainer.innerHTML = "";
  for (let d = 0; d < numDigits; d++) {
    const btn = document.createElement("button");
    btn.textContent = d;
    btn.addEventListener("click", () => selectDigit(d));
    digitButtonsContainer.appendChild(btn);
  }
}

init();

// --- Model picker ---
const modelButtonsContainer = document.getElementById("model-buttons");
modelButtonsContainer.addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-model]");
  if (!btn) return;

  const modelName = btn.dataset.model;
  setModel(modelName);

  // Update active button
  modelButtonsContainer.querySelectorAll("button").forEach((b) => {
    b.classList.toggle("active", b === btn);
  });

  // Reset UI state
  selectedIndex = null;
  document.getElementById("run-btn").disabled = true;
  document.getElementById("result").classList.add("hidden");
  document.getElementById("sample-thumbnails").innerHTML = "";

  // Tear down and rebuild
  teardownScene();
  await init();
});

// --- Digit picker ---
const digitButtonsContainer = document.getElementById("digit-buttons");

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

  // Derive image dimensions from input layer
  const inputLayer = layers[0];
  const side = inputLayer.cols;
  const pixelCount = inputLayer.size;

  for (const sample of samples) {
    const canvas = document.createElement("canvas");
    canvas.width = side;
    canvas.height = Math.ceil(pixelCount / side);
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(canvas.width, canvas.height);

    for (let i = 0; i < pixelCount; i++) {
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
  await animateFeedforward(layerMeshes, connectionMeshes, data.activations, layers, weightLayers);

  // Show result
  const maxProb = Math.max(...data.activations.probabilities);
  const pct = (maxProb * 100).toFixed(1);
  document.getElementById("result-text").textContent =
    `Predicted: ${data.prediction} (${pct}%)`;
  document.getElementById("result").classList.remove("hidden");

  btn.textContent = "Run Inference";
  btn.disabled = false;
});
