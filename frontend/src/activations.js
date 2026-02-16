import * as THREE from "three";
import { ANIMATION_LAYER_DELAY, ANIMATION_TWEEN_MS, SCAN_STEP_MS, SCAN_HIGHLIGHT_COLOR, CONV_KERNEL_SIZE } from "./constants.js";
import { resetAllLayers, resetConnections, neuronPosition } from "./network.js";

let animationGen = 0;
export function cancelAnimation() { animationGen++; }

// Dark-to-cyan gradient: near-black -> deep blue -> bright cyan -> white
// Uses sqrt curve so mid-range activations are more visible (most ReLU values cluster near 0)
const COLOR_STOPS = [
  { t: 0.0, r: 0.02, g: 0.02, b: 0.06 },
  { t: 0.3, r: 0.0, g: 0.15, b: 0.55 },
  { t: 0.65, r: 0.0, g: 0.8, b: 1.0 },
  { t: 1.0, r: 1.0, g: 1.0, b: 1.0 },
];

function activationToColor(value, maxVal) {
  const linear = maxVal > 0 ? Math.min(value / maxVal, 1) : 0;
  const t = Math.sqrt(linear);

  // Find the two stops we sit between
  let lo = COLOR_STOPS[0];
  let hi = COLOR_STOPS[COLOR_STOPS.length - 1];
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    if (t >= COLOR_STOPS[i].t && t <= COLOR_STOPS[i + 1].t) {
      lo = COLOR_STOPS[i];
      hi = COLOR_STOPS[i + 1];
      break;
    }
  }

  const f = hi.t === lo.t ? 0 : (t - lo.t) / (hi.t - lo.t);
  return [
    lo.r + (hi.r - lo.r) * f,
    lo.g + (hi.g - lo.g) * f,
    lo.b + (hi.b - lo.b) * f,
  ];
}

// Map pixel value (0-1) to grayscale RGB
function pixelToColor(value) {
  return [value, value, value];
}

// Color array for a layer given its activation values
function layerColors(layerName, values) {
  if (layerName === "input") {
    return values.map(pixelToColor);
  }
  const maxVal = Math.max(...values);
  return values.map((v) => activationToColor(v, maxVal));
}

function sleep(ms, gen) {
  return new Promise((resolve) => setTimeout(() => resolve(gen === animationGen), ms));
}

// Create a wireframe rectangle sized to cover a 3x3 kernel patch
function createScanWindow(spacing) {
  const size = CONV_KERNEL_SIZE * spacing;
  const half = size / 2;
  const vertices = new Float32Array([
    -half,  half, 0,
     half,  half, 0,
     half, -half, 0,
    -half, -half, 0,
  ]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
  const material = new THREE.LineBasicMaterial({ color: 0xffee33 });
  return new THREE.LineLoop(geometry, material);
}

// Animate a 3x3 kernel scanning across one output channel of a conv layer
async function scanConvLayer(scene, layerMeshes, srcLayer, dstLayer, dstActivationValues, gen) {
  // Source spatial dimensions
  const srcH = srcLayer.type === "input" ? srcLayer.rows : srcLayer.mapH;
  const srcW = srcLayer.type === "input" ? srcLayer.cols : srcLayer.mapW;

  // Destination spatial dimensions
  const dstH = dstLayer.mapH;
  const dstW = dstLayer.mapW;

  // Pool stride (conv+pool combined)
  const poolStride = srcH / dstH;

  const srcChannel = 0;
  const dstChannel = 0;

  // Create and add wireframe
  const scanWindow = createScanWindow(srcLayer.spacing);
  scene.add(scanWindow);

  const srcMesh = layerMeshes[srcLayer.name];
  const dstMesh = layerMeshes[dstLayer.name];

  const maxVal = Math.max(...dstActivationValues);

  const offsets = [-1, 0, 1];

  for (let r = 0; r < dstH; r++) {
    if (gen !== animationGen) break;
    for (let c = 0; c < dstW; c++) {
      if (gen !== animationGen) break;
      // Center position in source
      const centerRow = poolStride * r;
      const centerCol = poolStride * c;
      let centerFlatIdx;
      if (srcLayer.type === "input") {
        centerFlatIdx = centerRow * srcW + centerCol;
      } else {
        centerFlatIdx = srcChannel * srcH * srcW + centerRow * srcW + centerCol;
      }

      // Position wireframe at source center
      const centerPos = neuronPosition(srcLayer, centerFlatIdx);
      scanWindow.position.set(centerPos.x, centerPos.y, centerPos.z + 0.5);

      // Highlight source patch and save original colors
      const savedColors = [];
      for (const dr of offsets) {
        for (const dc of offsets) {
          const srcRow = centerRow + dr;
          const srcCol = centerCol + dc;
          if (srcRow >= 0 && srcRow < srcH && srcCol >= 0 && srcCol < srcW) {
            let flatIdx;
            if (srcLayer.type === "input") {
              flatIdx = srcRow * srcW + srcCol;
            } else {
              flatIdx = srcChannel * srcH * srcW + srcRow * srcW + srcCol;
            }
            const arr = srcMesh.instanceColor.array;
            const base = flatIdx * 3;
            savedColors.push({ flatIdx, r: arr[base], g: arr[base + 1], b: arr[base + 2] });
            arr[base] = SCAN_HIGHLIGHT_COLOR[0];
            arr[base + 1] = SCAN_HIGHLIGHT_COLOR[1];
            arr[base + 2] = SCAN_HIGHLIGHT_COLOR[2];
          }
        }
      }
      srcMesh.instanceColor.needsUpdate = true;

      // Light up destination neuron
      const dstFlatIdx = dstChannel * dstH * dstW + r * dstW + c;
      const color = activationToColor(dstActivationValues[dstFlatIdx], maxVal);
      const dstArr = dstMesh.instanceColor.array;
      const dstBase = dstFlatIdx * 3;
      dstArr[dstBase] = color[0];
      dstArr[dstBase + 1] = color[1];
      dstArr[dstBase + 2] = color[2];
      dstMesh.instanceColor.needsUpdate = true;

      const alive = await sleep(SCAN_STEP_MS, gen);
      if (!alive) {
        // Restore source patch before bailing
        for (const saved of savedColors) {
          const base = saved.flatIdx * 3;
          const arr = srcMesh.instanceColor.array;
          arr[base] = saved.r;
          arr[base + 1] = saved.g;
          arr[base + 2] = saved.b;
        }
        srcMesh.instanceColor.needsUpdate = true;
        scene.remove(scanWindow);
        scanWindow.geometry.dispose();
        scanWindow.material.dispose();
        return;
      }

      // Restore source patch colors
      for (const saved of savedColors) {
        const base = saved.flatIdx * 3;
        const arr = srcMesh.instanceColor.array;
        arr[base] = saved.r;
        arr[base + 1] = saved.g;
        arr[base + 2] = saved.b;
      }
      srcMesh.instanceColor.needsUpdate = true;
    }
  }

  // Cleanup wireframe
  scene.remove(scanWindow);
  scanWindow.geometry.dispose();
  scanWindow.material.dispose();
}

// Tween connection opacity from 0 to target over duration ms
function tweenOpacity(lineMesh, target, duration, gen) {
  return new Promise((resolve) => {
    const start = performance.now();
    const initial = lineMesh.material.opacity;
    function step(now) {
      if (gen !== animationGen) { resolve(); return; }
      const t = Math.min((now - start) / duration, 1);
      lineMesh.material.opacity = initial + (target - initial) * t;
      if (t < 1) {
        requestAnimationFrame(step);
      } else {
        resolve();
      }
    }
    requestAnimationFrame(step);
  });
}

// Tween layer colors from current to target over duration ms
function tweenLayerColors(layerMeshes, layer, targetColors, duration, gen) {
  return new Promise((resolve) => {
    const mesh = layerMeshes[layer.name];
    const startColors = [];

    // Read current colors
    for (let i = 0; i < layer.size; i++) {
      const arr = mesh.instanceColor.array;
      startColors.push([arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]]);
    }

    const startTime = performance.now();

    function step(now) {
      if (gen !== animationGen) { resolve(); return; }
      const t = Math.min((now - startTime) / duration, 1);
      const eased = t * t * (3 - 2 * t); // smoothstep
      for (let i = 0; i < layer.size; i++) {
        const sr = startColors[i][0],
          sg = startColors[i][1],
          sb = startColors[i][2];
        const [tr, tg, tb] = targetColors[i];
        mesh.instanceColor.array[i * 3] = sr + (tr - sr) * eased;
        mesh.instanceColor.array[i * 3 + 1] = sg + (tg - sg) * eased;
        mesh.instanceColor.array[i * 3 + 2] = sb + (tb - sb) * eased;
      }
      mesh.instanceColor.needsUpdate = true;
      if (t < 1) {
        requestAnimationFrame(step);
      } else {
        resolve();
      }
    }
    requestAnimationFrame(step);
  });
}

// Build a lookup from layer index to its weightLayer entry (if any)
function buildWeightLookup(layers, weightLayers) {
  const lookup = {};
  for (const wl of weightLayers) {
    lookup[wl.dstIdx] = wl;
  }
  return lookup;
}

// Run the full feedforward animation sequence
export async function animateFeedforward(scene, layerMeshes, connectionMeshes, activations, layers, weightLayers) {
  const gen = ++animationGen;

  resetAllLayers(layerMeshes, layers);
  resetConnections(connectionMeshes, weightLayers);

  const weightLookup = buildWeightLookup(layers, weightLayers);

  // Step 1: Light up input layer
  const inputLayer = layers[0];
  const inputColors = layerColors(inputLayer.name, activations.input);
  await tweenLayerColors(layerMeshes, inputLayer, inputColors, ANIMATION_TWEEN_MS, gen);
  if (gen !== animationGen) return null;

  // Steps 2+: For each subsequent layer, fade in connections (if any) then color neurons
  for (let i = 1; i < layers.length; i++) {
    const alive = await sleep(ANIMATION_LAYER_DELAY, gen);
    if (!alive) return null;

    // Fade in connections if this layer has weight connections
    const wl = weightLookup[i];
    if (wl && connectionMeshes[wl.weightKey]) {
      await tweenOpacity(connectionMeshes[wl.weightKey], 0.6, ANIMATION_TWEEN_MS, gen);
      if (gen !== animationGen) return null;
    }

    // Color neurons
    const layer = layers[i];
    const colors = layerColors(layer.name, activations[layer.name]);

    // Scanning animation for conv layers
    if (layer.type === "conv2d" && wl) {
      const srcLayer = layers[wl.srcIdx];
      await scanConvLayer(scene, layerMeshes, srcLayer, layer, activations[layer.name], gen);
      if (gen !== animationGen) return null;
    }

    // Tween all neurons to final colors (fills remaining channels after scan, or full layer for non-conv)
    await tweenLayerColors(layerMeshes, layer, colors, ANIMATION_TWEEN_MS, gen);
    if (gen !== animationGen) return null;
  }

  return activations;
}
