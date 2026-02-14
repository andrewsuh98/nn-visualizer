export const LAYERS = [
  { name: "input", size: 784, cols: 28, rows: 28, z: 0 },
  { name: "fc1_relu", size: 128, cols: 16, rows: 8, z: 30 },
  { name: "fc2_relu", size: 64, cols: 8, rows: 8, z: 60 },
  { name: "output", size: 10, cols: 10, rows: 1, z: 90 },
];

export const WEIGHT_LAYERS = ["fc1", "fc2", "fc3"];

export const NEURON_RADIUS = 0.35;
export const NEURON_SPACING = 1.2;

export const ANIMATION_LAYER_DELAY = 800;
export const ANIMATION_TWEEN_MS = 300;

export const BG_COLOR = 0x1a1a2e;

export const API_BASE = "http://localhost:8000";
