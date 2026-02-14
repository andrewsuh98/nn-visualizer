export const NEURON_RADIUS = 0.35;
export const NEURON_SPACING = 1.2;
export const NEURON_SPACING_CONV = 0.8;

export const ANIMATION_LAYER_DELAY = 800;
export const ANIMATION_TWEEN_MS = 300;

export const BG_COLOR = 0x1a1a2e;

export const API_BASE = "http://localhost:8000";

export const Z_SPACING = 35;

// Build dynamic layer config from architecture response.
// Each layer gets grid dimensions (cols/rows) and a z-position.
export function buildLayerConfig(archData) {
  const layers = [];
  const weightLayers = [];
  const totalLayers = archData.layers.length;

  for (let i = 0; i < totalLayers; i++) {
    const arch = archData.layers[i];
    const shape = arch.shape;
    const type = arch.type;

    let cols, rows, size, spacing, displayName;
    let mapW = 0, mapH = 0, gridCols = 0, gridRows = 0, mapGap = 0;

    if (type === "input") {
      // 2D grid: 28 cols x 28 rows
      cols = shape[2]; // W
      rows = shape[1]; // H
      size = shape[1] * shape[2];
      spacing = NEURON_SPACING;
      displayName = `Input (${cols}x${rows})`;
    } else if (type === "conv2d") {
      // Tiled 2D feature maps
      const C = shape[0], H = shape[1], W = shape[2];
      mapW = W;
      mapH = H;
      size = C * H * W;
      // Arrange C maps in a grid -- prefer wider than tall
      gridCols = Math.ceil(Math.sqrt(C));
      gridRows = Math.ceil(C / gridCols);
      mapGap = 2; // gap in neuron-spacing units between maps
      // Total extent: gridCols maps of width W with gaps between
      cols = gridCols * W + (gridCols - 1) * mapGap;
      rows = gridRows * H + (gridRows - 1) * mapGap;
      spacing = NEURON_SPACING_CONV;
      displayName = `Conv (${C}x${H}x${W})`;
    } else {
      // linear: single row
      size = shape[0];
      cols = size;
      rows = 1;
      spacing = NEURON_SPACING;
      if (arch.name === "logits") {
        displayName = `Output (${size}, Softmax)`;
      } else {
        displayName = `Hidden (${size}, ReLU)`;
      }
    }

    // Activation key: for logits layer, we also need probabilities
    let activationKey;
    if (type === "input") {
      activationKey = "input";
    } else if (arch.name === "logits") {
      activationKey = "probabilities";
    } else {
      activationKey = arch.name;
    }

    layers.push({
      name: activationKey,
      size,
      cols,
      rows,
      z: i * Z_SPACING,
      spacing,
      displayName,
      type,
      shape,
      // Conv-specific layout info
      mapW,
      mapH,
      gridCols,
      gridRows,
      mapGap,
      weightKey: arch.weight_key,
    });

    // Only layers with a weight_key get weight connections
    if (arch.weight_key) {
      weightLayers.push({
        weightKey: arch.weight_key,
        srcIdx: i - 1,
        dstIdx: i,
      });
    }
  }

  return { layers, weightLayers };
}
