# Neural Network Visualizer

An interactive 3D visualization of neural networks processing handwritten digits. Watch activations flow layer-by-layer through MLP and CNN architectures trained on MNIST.

Built with Three.js, FastAPI, and PyTorch.

<!-- TODO: replace with actual gif path once recorded -->
![Demo](demo.gif)

## How It Works

1. **Pick a digit** (0--9) to load sample images from MNIST
2. **Select a sample** thumbnail
3. **Run Inference** to animate the feedforward pass:
   - Input pixels light up in grayscale
   - Each hidden layer activates sequentially (dark-to-cyan gradient)
   - Weight connections fade in between layers (blue = positive, red = negative)
   - Output layer shows class probabilities; prediction and confidence are displayed

Switch between an MLP and a CNN at any time -- the visualization rebuilds automatically to match the architecture.

## Features

- **3D network rendering** -- neurons as spheres, weight connections as colored lines, all in an interactive orbit-controlled scene
- **Animated inference** -- activations propagate through each layer in sequence
- **Multiple architectures** -- MLP (3 fully-connected layers) and CNN (2 conv layers + fully-connected)
- **Data-driven layout** -- visualization adapts automatically to any architecture; conv layers render as tiled feature maps, linear layers as rows

## Running Locally

Requires Python 3.14+ with [uv](https://docs.astral.sh/uv/) and Node.js 18+.

```bash
uv sync                              # Python deps
cd frontend && npm install && cd ..   # Frontend deps

# Train models (skip if models/ already has weights)
python main.py --model mlp --epochs 10
python main.py --model cnn --epochs 10
```

Then start both servers:

```bash
uvicorn server:app --reload           # Backend on :8000
cd frontend && npm run dev            # Frontend on :5173
```

## Project Structure

```
frontend/src/       Three.js + vanilla JS
  main.js           Orchestration, UI events, camera framing
  scene.js          Three.js scene, camera, lighting, controls
  network.js        Neuron and connection mesh construction
  activations.js    Feedforward animation sequencing
  constants.js      Shared config and architecture-to-layout mapping
  api.js            FastAPI client

server.py           FastAPI backend (architecture, samples, inference, weights)
main.py             PyTorch model definitions (MLP, CNN) and training
```
