import "./style.css";
import * as THREE from "three";
import { createScene } from "./scene.js";
import { buildNeurons, buildConnections, resetAllLayers, resetConnections } from "./network.js";
import { fetchArchitecture, fetchInferenceDraw, fetchWeights, setModel } from "./api.js";
import { animateFeedforward } from "./activations.js";
import { buildLayerConfig, Z_SPACING, CAMERA_TWEEN_MS } from "./constants.js";

let layerMeshes = null;
let connectionMeshes = null;
let layers = null;
let weightLayers = null;
let currentActivations = null;
let pinnedNeuron = null;
let isFirstInit = true;
let cameraTweenId = null;
let defaultCameraPos = null;
let defaultCameraTarget = null;

// Initialize Three.js scene
const container = document.getElementById("canvas-container");
const { scene, camera, renderer, controls } = createScene(container);

// --- Raycaster + tooltip ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltipEl = document.getElementById("neuron-tooltip");

function findNeuronAtMouse() {
  if (!layerMeshes || !layers) return null;
  raycaster.setFromCamera(mouse, camera);

  let closest = null;
  for (const layer of layers) {
    const mesh = layerMeshes[layer.name];
    if (!mesh) continue;
    const hits = raycaster.intersectObject(mesh);
    if (hits.length > 0) {
      const hit = hits[0];
      if (!closest || hit.distance < closest.distance) {
        closest = {
          layerName: layer.name,
          displayName: layer.displayName,
          instanceId: hit.instanceId,
          distance: hit.distance,
        };
      }
    }
  }
  return closest;
}

function showTooltip(layerName, displayName, instanceId, clientX, clientY) {
  let activationText = "N/A";
  if (currentActivations && currentActivations[layerName]) {
    const vals = currentActivations[layerName];
    if (instanceId < vals.length) {
      activationText = vals[instanceId].toFixed(4);
    }
  }

  tooltipEl.innerHTML =
    `<strong>${displayName}</strong><br>` +
    `Neuron: ${instanceId}<br>` +
    `Activation: ${activationText}`;
  tooltipEl.classList.remove("hidden");

  // Position near cursor, clamped to viewport
  const pad = 15;
  let x = clientX + pad;
  let y = clientY + pad;
  const rect = tooltipEl.getBoundingClientRect();
  if (x + rect.width > window.innerWidth) x = clientX - pad - rect.width;
  if (y + rect.height > window.innerHeight) y = clientY - pad - rect.height;
  tooltipEl.style.left = x + "px";
  tooltipEl.style.top = y + "px";
}

function hideTooltip() {
  tooltipEl.classList.add("hidden");
  pinnedNeuron = null;
}

renderer.domElement.addEventListener("mousemove", (e) => {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  if (pinnedNeuron) return; // tooltip is pinned, don't update on hover

  const hit = findNeuronAtMouse();
  if (hit) {
    showTooltip(hit.layerName, hit.displayName, hit.instanceId, e.clientX, e.clientY);
  } else {
    hideTooltip();
  }
});

renderer.domElement.addEventListener("click", (e) => {
  const hit = findNeuronAtMouse();
  if (hit) {
    pinnedNeuron = { ...hit };
    showTooltip(hit.layerName, hit.displayName, hit.instanceId, e.clientX, e.clientY);
  } else {
    hideTooltip();
  }
});

function tweenCamera(targetPos, targetLookAt, duration) {
  if (cameraTweenId !== null) {
    cancelAnimationFrame(cameraTweenId);
    cameraTweenId = null;
  }

  const startPos = camera.position.clone();
  const startTarget = controls.target.clone();
  const startTime = performance.now();

  return new Promise((resolve) => {
    function tick() {
      const elapsed = performance.now() - startTime;
      const raw = Math.min(elapsed / duration, 1);
      const t = 1 - (1 - raw) * (1 - raw); // ease-out quadratic

      camera.position.lerpVectors(startPos, targetPos, t);
      controls.target.lerpVectors(startTarget, targetLookAt, t);
      controls.update();

      if (raw < 1) {
        cameraTweenId = requestAnimationFrame(tick);
      } else {
        cameraTweenId = null;
        resolve();
      }
    }
    cameraTweenId = requestAnimationFrame(tick);
  });
}

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
  currentActivations = null;
  pinnedNeuron = null;
  hideTooltip();

  // Fade out legend (content swapped in init)
  document.getElementById("legend-layers").classList.add("fade-out");
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

  // Dynamically position camera to frame the entire network
  const zExtent = (layers.length - 1) * Z_SPACING;
  const zCenter = zExtent / 2;

  // Camera viewing angle in the xz plane (45 deg from z-axis)
  const angle = Math.PI / 4;
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);

  const vFov = (camera.fov * Math.PI) / 180;
  const aspect = camera.aspect;
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * aspect);

  // For each layer, compute the minimum camera distance so it fits within
  // the frustum. Layers with positive dz (closer to camera) need extra
  // distance because they subtend a larger angle.
  const tanH = Math.tan(hFov / 2);
  const tanV = Math.tan(vFov / 2);
  let minDist = 0;
  for (const layer of layers) {
    const halfW = (layer.cols * layer.spacing) / 2;
    const halfH = (layer.rows * layer.spacing) / 2;
    const dz = layer.z - zCenter; // signed: positive = closer to camera
    const extentH = Math.abs(dz) * sinA + halfW * cosA;
    const dH = extentH / tanH + dz * cosA;
    const dV = halfH / tanV + dz * cosA;
    const d = Math.max(dH, dV);
    if (d > minDist) minDist = d;
  }

  const padding = 1.00;
  const dist = minDist * padding;

  // Position camera at the computed distance along the quarter-angle direction
  const camX = -dist * sinA;
  const camZ = zCenter + dist * cosA;
  const camY = dist * 0.25;

  const newTarget = new THREE.Vector3(0, 0, zCenter);
  const newPos = new THREE.Vector3(camX, camY, camZ);

  defaultCameraPos = newPos.clone();
  defaultCameraTarget = newTarget.clone();

  if (isFirstInit) {
    camera.position.copy(newPos);
    controls.target.copy(newTarget);
    controls.update();
    isFirstInit = false;
  } else {
    tweenCamera(newPos, newTarget, CAMERA_TWEEN_MS);
  }

  // Build legend dynamically (crossfade on model switch)
  const legendContainer = document.getElementById("legend-layers");
  legendContainer.innerHTML = "";
  for (const layer of layers) {
    const item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML = `<span class="legend-dot"></span>${layer.displayName}`;
    legendContainer.appendChild(item);
  }
  // Force reflow so the browser registers opacity:0 before removing the class
  legendContainer.offsetHeight;
  legendContainer.classList.remove("fade-out");
}

init();

// --- Model picker ---
const modelButtonsContainer = document.getElementById("model-buttons");
modelButtonsContainer.addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-model]");
  if (!btn) return;

  const modelName = btn.dataset.model;
  setModel(modelName);

  modelButtonsContainer.querySelectorAll("button").forEach((b) => {
    b.classList.toggle("active", b === btn);
  });

  document.getElementById("result").classList.add("hidden");
  document.getElementById("replay-btn").disabled = true;

  teardownScene();
  await init();
});

// --- Drawing canvas ---
const drawCanvas = document.getElementById("draw-canvas");
const drawCtx = drawCanvas.getContext("2d");
let isDrawing = false;
let debounceTimer = null;

// Fill black initially
drawCtx.fillStyle = "#000";
drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);

function getDrawPos(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY,
  };
}

function startDraw(e) {
  e.preventDefault();
  isDrawing = true;
  if (debounceTimer) {
    clearTimeout(debounceTimer);
    debounceTimer = null;
  }
  const pos = getDrawPos(e);
  drawCtx.beginPath();
  drawCtx.moveTo(pos.x, pos.y);
  // Draw a dot for single clicks/taps
  drawCtx.lineTo(pos.x + 0.1, pos.y + 0.1);
  drawCtx.strokeStyle = "#fff";
  drawCtx.lineWidth = 14;
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.stroke();
}

function moveDraw(e) {
  if (!isDrawing) return;
  e.preventDefault();
  const pos = getDrawPos(e);
  drawCtx.lineTo(pos.x, pos.y);
  drawCtx.stroke();
}

function endDraw(e) {
  if (!isDrawing) return;
  e.preventDefault();
  isDrawing = false;
  // Debounce: run inference 600ms after pen lifts
  debounceTimer = setTimeout(runDrawInference, 600);
}

drawCanvas.addEventListener("pointerdown", startDraw);
drawCanvas.addEventListener("pointermove", moveDraw);
drawCanvas.addEventListener("pointerup", endDraw);
drawCanvas.addEventListener("pointerleave", endDraw);

async function runDrawInference() {
  if (!connectionMeshes) return;

  // Downsample to 28x28
  const offscreen = document.createElement("canvas");
  offscreen.width = 28;
  offscreen.height = 28;
  const offCtx = offscreen.getContext("2d");
  offCtx.drawImage(drawCanvas, 0, 0, 28, 28);
  const imageData = offCtx.getImageData(0, 0, 28, 28);
  const pixels = [];
  for (let i = 0; i < 784; i++) {
    pixels.push(imageData.data[i * 4] / 255); // red channel
  }

  document.getElementById("result").classList.add("hidden");
  hideTooltip();

  const data = await fetchInferenceDraw(pixels);
  await animateFeedforward(layerMeshes, connectionMeshes, data.activations, layers, weightLayers);
  currentActivations = data.activations;

  const maxProb = Math.max(...data.activations.probabilities);
  const pct = (maxProb * 100).toFixed(1);
  const resultText = document.getElementById("result-text");
  resultText.textContent = `Predicted: ${data.prediction} (${pct}%)`;
  // Red (<40%) -> Yellow (~75%) -> Green (>95%)
  const t = Math.max(0, Math.min(1, (maxProb - 0.4) / 0.55));
  const r = t < 0.5 ? 255 : Math.round(255 * (1 - t) * 2);
  const g = t < 0.5 ? Math.round(255 * t * 2) : 255;
  resultText.style.color = `rgb(${r}, ${g}, 50)`;
  document.getElementById("result").classList.remove("hidden");

  document.getElementById("replay-btn").disabled = false;
}

// --- Clear button ---
document.getElementById("clear-btn").addEventListener("click", () => {
  drawCtx.fillStyle = "#000";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  if (debounceTimer) {
    clearTimeout(debounceTimer);
    debounceTimer = null;
  }
  if (layerMeshes && layers) resetAllLayers(layerMeshes, layers);
  if (connectionMeshes && weightLayers) resetConnections(connectionMeshes, weightLayers);
  currentActivations = null;
  document.getElementById("result").classList.add("hidden");
  document.getElementById("replay-btn").disabled = true;
  hideTooltip();
});

// --- Replay button ---
document.getElementById("replay-btn").addEventListener("click", async () => {
  if (!currentActivations || !layerMeshes || !connectionMeshes) return;
  document.getElementById("replay-btn").disabled = true;
  resetAllLayers(layerMeshes, layers);
  resetConnections(connectionMeshes, weightLayers);
  await animateFeedforward(layerMeshes, connectionMeshes, currentActivations, layers, weightLayers);
  document.getElementById("replay-btn").disabled = false;
});

// --- Reset camera button ---
const resetBtn = document.getElementById("reset-camera");
const CAMERA_THRESHOLD = 0.5;

controls.addEventListener("change", () => {
  if (!defaultCameraPos) return;
  const posDist = camera.position.distanceTo(defaultCameraPos);
  const targetDist = controls.target.distanceTo(defaultCameraTarget);
  if (posDist > CAMERA_THRESHOLD || targetDist > CAMERA_THRESHOLD) {
    resetBtn.classList.remove("hidden");
  } else {
    resetBtn.classList.add("hidden");
  }
});

resetBtn.addEventListener("click", () => {
  if (!defaultCameraPos) return;
  tweenCamera(defaultCameraPos.clone(), defaultCameraTarget.clone(), CAMERA_TWEEN_MS);
  resetBtn.classList.add("hidden");
});
