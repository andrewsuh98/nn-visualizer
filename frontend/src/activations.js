import { LAYERS, WEIGHT_LAYERS, ANIMATION_LAYER_DELAY, ANIMATION_TWEEN_MS } from "./constants.js";
import { resetAllLayers, resetConnections } from "./network.js";

// Dark-to-cyan gradient: near-black -> deep blue -> bright cyan -> white
// Uses sqrt curve so mid-range activations are more visible (most ReLU values cluster near 0)
const COLOR_STOPS = [
  { t: 0.0, r: 0.02, g: 0.02, b: 0.05 },
  { t: 0.33, r: 0.0, g: 0.1, b: 0.4 },
  { t: 0.66, r: 0.0, g: 0.7, b: 0.9 },
  { t: 1.0, r: 0.85, g: 1.0, b: 1.0 },
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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Tween connection opacity from 0 to target over duration ms
function tweenOpacity(lineMesh, target, duration) {
  return new Promise((resolve) => {
    const start = performance.now();
    const initial = lineMesh.material.opacity;
    function step(now) {
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
function tweenLayerColors(layerMeshes, layerName, targetColors, duration) {
  return new Promise((resolve) => {
    const mesh = layerMeshes[layerName];
    const layer = LAYERS.find((l) => l.name === layerName);
    const startColors = [];

    // Read current colors
    for (let i = 0; i < layer.size; i++) {
      // InstancedMesh stores colors in the instanceColor buffer
      const arr = mesh.instanceColor.array;
      startColors.push([arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]]);
    }

    const startTime = performance.now();

    function step(now) {
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

// Run the full feedforward animation sequence
export async function animateFeedforward(layerMeshes, connectionMeshes, activations) {
  resetAllLayers(layerMeshes);
  resetConnections(connectionMeshes);

  // Step 1: Light up input layer
  const inputColors = layerColors("input", activations.input);
  await tweenLayerColors(layerMeshes, "input", inputColors, ANIMATION_TWEEN_MS);

  // Steps 2-4: For each subsequent layer, fade in connections then color neurons
  const layerKeys = ["fc1_relu", "fc2_relu", "output"];
  const activationKeys = ["fc1_relu", "fc2_relu", "probabilities"];

  for (let i = 0; i < layerKeys.length; i++) {
    await sleep(ANIMATION_LAYER_DELAY);

    // Fade in connections
    const weightName = WEIGHT_LAYERS[i];
    await tweenOpacity(connectionMeshes[weightName], 0.6, ANIMATION_TWEEN_MS);

    // Color neurons
    const colors = layerColors(layerKeys[i], activations[activationKeys[i]]);
    await tweenLayerColors(layerMeshes, layerKeys[i], colors, ANIMATION_TWEEN_MS);
  }

  return activations;
}
