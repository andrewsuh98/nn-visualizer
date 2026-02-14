import random

import torch
import torch.nn as nn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from torchvision import datasets, transforms

from main import MLP

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load model
model = MLP()
model.load_state_dict(torch.load("models/mnist_mlp.pth", map_location="cpu", weights_only=True))
model.eval()

# Discover Linear layers in order
linear_layers: list[tuple[str, nn.Linear]] = [
    (name, module)
    for name, module in model.named_modules()
    if isinstance(module, nn.Linear)
]

# Two copies of test set: raw (0-1) for display, normalized for inference
raw_transform = transforms.ToTensor()
norm_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

raw_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=raw_transform)
norm_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=norm_transform)

# Index samples by digit for fast lookup
digit_indices: dict[int, list[int]] = {d: [] for d in range(10)}
for i, (_, label) in enumerate(raw_dataset):
    digit_indices[label].append(i)


def get_activations(x_normalized: torch.Tensor) -> dict:
    """Run forward pass with hooks on each Linear layer to capture activations."""
    captured: dict[str, torch.Tensor] = {}

    hooks = []
    for name, layer in linear_layers:
        def make_hook(n):
            def hook_fn(module, input, output):
                captured[n] = output
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        flat = x_normalized.view(1, -1)
        logits = model(flat)
        probs = torch.softmax(logits, dim=1)

    for h in hooks:
        h.remove()

    result = {}
    for i, (name, layer) in enumerate(linear_layers):
        is_last = i == len(linear_layers) - 1
        raw_out = captured[name].squeeze()
        if is_last:
            result["logits"] = raw_out.tolist()
            result["probabilities"] = probs.squeeze().tolist()
        else:
            # Apply ReLU to match what the model does internally
            result[f"{name}_relu"] = torch.relu(raw_out).tolist()

    return result


@app.get("/api/architecture")
def architecture():
    layers = []
    # Input layer: in_features of the first Linear layer
    input_size = linear_layers[0][1].in_features
    layers.append({"name": "input", "size": input_size, "type": "input"})

    for i, (name, module) in enumerate(linear_layers):
        is_last = i == len(linear_layers) - 1
        layers.append({
            "name": name,
            "size": module.out_features,
            "type": "linear",
            "activation": "softmax" if is_last else "relu",
        })

    return {"layers": layers}


@app.get("/api/samples")
def samples(digit: int = Query(ge=0, le=9), count: int = Query(default=10, ge=1, le=50)):
    indices = random.sample(digit_indices[digit], min(count, len(digit_indices[digit])))
    results = []
    for idx in indices:
        img, label = raw_dataset[idx]
        results.append({
            "index": idx,
            "label": int(label),
            "pixels": img.squeeze().flatten().tolist(),
        })
    return results


@app.get("/api/inference/{index}")
def inference(index: int):
    raw_img, label = raw_dataset[index]
    norm_img, _ = norm_dataset[index]

    acts = get_activations(norm_img)
    prediction = int(torch.tensor(acts["logits"]).argmax())

    return {
        "label": int(label),
        "prediction": prediction,
        "activations": {
            "input": raw_img.squeeze().flatten().tolist(),
            **acts,
        },
    }


@app.get("/api/weights")
def weights():
    top_k = 5
    result = {}
    for name, layer in linear_layers:
        w = layer.weight.data  # shape: (out_features, in_features)
        connections = []
        for dst in range(layer.out_features):
            row = w[dst]
            _, top_indices = row.abs().topk(top_k)
            for src_idx in top_indices.tolist():
                connections.append({
                    "src": src_idx,
                    "dst": dst,
                    "weight": float(row[src_idx]),
                })
        result[name] = connections
    return result
