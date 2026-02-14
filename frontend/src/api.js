import { API_BASE } from "./constants.js";

let currentModel = "mlp";

export function setModel(name) {
  currentModel = name;
}

export function getModel() {
  return currentModel;
}

export async function fetchArchitecture() {
  const res = await fetch(`${API_BASE}/api/architecture?model=${currentModel}`);
  return res.json();
}

export async function fetchSamples(digit, count = 10) {
  const res = await fetch(`${API_BASE}/api/samples?digit=${digit}&count=${count}`);
  return res.json();
}

export async function fetchInference(index) {
  const res = await fetch(`${API_BASE}/api/inference/${index}?model=${currentModel}`);
  return res.json();
}

export async function fetchWeights() {
  const res = await fetch(`${API_BASE}/api/weights?model=${currentModel}`);
  return res.json();
}
