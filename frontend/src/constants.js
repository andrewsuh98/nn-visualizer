export const NEURON_RADIUS = 0.35;
export const NEURON_SPACING = 1.2;

export const ANIMATION_LAYER_DELAY = 800;
export const ANIMATION_TWEEN_MS = 300;

export const BG_COLOR = 0x1a1a2e;

export const API_BASE = "http://localhost:8000";

// Build dynamic layer config from architecture response.
// Each layer gets grid dimensions (cols/rows) and a z-position.
export function buildLayerConfig(archData) {
  const layers = [];
  const weightLayers = [];
  const totalLayers = archData.layers.length;
  const zSpacing = 30;

  for (let i = 0; i < totalLayers; i++) {
    const arch = archData.layers[i];
    const size = arch.size;

    let cols, rows;
    if (arch.type === "input") {
      // Special-case: assume square input (e.g. 28x28 for 784)
      const side = Math.round(Math.sqrt(size));
      cols = side;
      rows = Math.ceil(size / side);
    } else {
      // Find roughly square layout
      cols = Math.ceil(Math.sqrt(size));
      rows = Math.ceil(size / cols);
    }

    // Build the activation key used in the activations response
    let activationKey;
    if (arch.type === "input") {
      activationKey = "input";
    } else if (arch.activation === "softmax") {
      activationKey = "probabilities";
    } else {
      activationKey = `${arch.name}_relu`;
    }

    // Build the display name for the layer
    let displayName;
    if (arch.type === "input") {
      displayName = `Input (${cols}x${rows} = ${size})`;
    } else if (arch.activation === "softmax") {
      displayName = `Output (${size}, Softmax)`;
    } else {
      displayName = `Hidden (${size}, ReLU)`;
    }

    layers.push({
      name: activationKey,
      size,
      cols,
      rows,
      z: i * zSpacing,
      displayName,
    });

    // All non-input layers have a weight layer connecting from previous
    if (arch.type !== "input") {
      weightLayers.push(arch.name);
    }
  }

  return { layers, weightLayers };
}
