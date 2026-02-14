# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A 3D neural network visualizer: PyTorch models (MLP & CNN) on MNIST with a Three.js frontend that animates feedforward inference layer-by-layer.

## Development Commands

```bash
# Backend (from project root)
uvicorn server:app --reload          # FastAPI on http://localhost:8000

# Frontend (from frontend/)
npm run dev                          # Vite dev server with HMR
npm run build                        # Production build to dist/

# Training (optional -- pretrained weights already exist in models/)
python main.py --model mlp --epochs 10
python main.py --model cnn --epochs 10
```

Both servers must run simultaneously. Frontend expects backend at `http://localhost:8000` (configured in `frontend/src/constants.js`).

## Architecture

**Backend** (`server.py`): Loads both pretrained models at startup into a `MODELS` dict. Four endpoints:
- `/api/architecture` -- layer metadata (name, type, shape) from model's `get_architecture()`
- `/api/samples` -- random MNIST images for a digit
- `/api/inference/{index}` -- runs `forward_viz()` returning prediction + all intermediate activations
- `/api/weights` -- top-5 connections per output neuron by weight magnitude

**Models** (`main.py`): MLP and CNN classes. Each implements `forward_viz()` (returns intermediate activations for visualization) and `get_architecture()` (returns layer metadata for dynamic UI generation).

**Frontend** (`frontend/src/`): Vanilla JS + Three.js, no framework.
- `main.js` -- orchestration: init, model switching, digit/sample selection, camera framing
- `scene.js` -- Three.js scene, camera, OrbitControls, lighting
- `network.js` -- builds InstancedMesh neurons and LineSegments connections; `neuronPosition()` handles both grid (linear) and tiled feature-map (conv2d) layouts
- `activations.js` -- sequential tweened animation of activations through layers
- `constants.js` -- shared config (spacing, radii, timing, colors) and `buildLayerConfig()` which translates architecture response into layer objects with grid dimensions and z-positions
- `api.js` -- fetch wrappers; `setModel()` switches the active model for all subsequent requests

**Data flow**: User picks digit -> picks sample thumbnail -> clicks Run Inference -> frontend fetches activations -> `animateFeedforward()` lights up layers sequentially with tweened color transitions.

## Key Conventions

- Frontend visualization is fully data-driven from `get_architecture()` -- no hardcoded layer counts or sizes
- Scene teardown/rebuild happens on model switch (not incremental updates)
- Camera distance is computed dynamically from frustum geometry to frame any model
- Neuron colors use a sqrt curve (`t = sqrt(v/max)`) to spread the dark-to-cyan gradient across ReLU-skewed activation distributions
- Weight connections show only top-5 per neuron; blue = positive, red = negative
- Python deps managed with `uv`; frontend deps with npm
