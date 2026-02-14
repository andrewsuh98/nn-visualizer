import { API_BASE } from "./constants.js";

export async function fetchSamples(digit, count = 10) {
  const res = await fetch(`${API_BASE}/api/samples?digit=${digit}&count=${count}`);
  return res.json();
}

export async function fetchInference(index) {
  const res = await fetch(`${API_BASE}/api/inference/${index}`);
  return res.json();
}

export async function fetchWeights() {
  const res = await fetch(`${API_BASE}/api/weights`);
  return res.json();
}
