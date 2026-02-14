import random

import torch
import torch.nn as nn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from torchvision import datasets, transforms

from main import MLP, CNN

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load both models into a registry
MODELS = {}
for name, cls, path in [
    ("mlp", MLP, "models/mnist_mlp.pth"),
    ("cnn", CNN, "models/mnist_cnn.pth"),
]:
    m = cls()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    MODELS[name] = m

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


def get_model(model_name: str):
    return MODELS[model_name]


@app.get("/api/architecture")
def architecture(model: str = Query(default="mlp")):
    m = get_model(model)
    return {"layers": m.get_architecture()}


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
def inference(index: int, model: str = Query(default="mlp")):
    m = get_model(model)
    raw_img, label = raw_dataset[index]
    norm_img, _ = norm_dataset[index]

    with torch.no_grad():
        # Shape [1, 1, 28, 28] -- MLP's forward_viz flattens internally
        x = norm_img.unsqueeze(0)
        logits, intermediates = m.forward_viz(x)
        probs = torch.softmax(logits, dim=1)

    prediction = int(logits.argmax(dim=1).item())

    activations = {"input": raw_img.squeeze().flatten().tolist()}
    for key, tensor in intermediates.items():
        activations[key] = tensor.squeeze().flatten().tolist()
    activations["probabilities"] = probs.squeeze().tolist()

    return {
        "label": int(label),
        "prediction": prediction,
        "activations": activations,
    }


@app.get("/api/weights")
def weights(model: str = Query(default="mlp")):
    m = get_model(model)
    top_k = 5
    result = {}

    # Build weight_key -> (src_shape, dst_shape) from architecture
    arch = m.get_architecture()
    key_to_shapes = {}
    for i, layer in enumerate(arch):
        wk = layer.get("weight_key")
        if wk and i > 0:
            key_to_shapes[wk] = (arch[i - 1]["shape"], layer["shape"])

    for name, module in m.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data  # shape: (out_features, in_features)
            connections = []
            for dst in range(module.out_features):
                row = w[dst]
                _, top_indices = row.abs().topk(top_k)
                for src_idx in top_indices.tolist():
                    connections.append({
                        "src": src_idx,
                        "dst": dst,
                        "weight": float(row[src_idx]),
                    })
            result[name] = connections
        elif isinstance(module, nn.Conv2d):
            w = module.weight.data  # shape: (out_ch, in_ch, kH, kW)
            out_ch, in_ch = w.shape[:2]
            importance = w.abs().sum(dim=(2, 3))  # shape: (out_ch, in_ch)
            signed = w.sum(dim=(2, 3))  # shape: (out_ch, in_ch)

            src_shape = key_to_shapes[name][0]  # e.g. [1, 28, 28] or [8, 14, 14]
            dst_shape = key_to_shapes[name][1]  # e.g. [8, 14, 14] or [16, 7, 7]
            src_h, src_w = src_shape[1], src_shape[2]
            dst_h, dst_w = dst_shape[1], dst_shape[2]

            connections = []
            k = min(top_k, in_ch)
            for dst_c in range(out_ch):
                _, top_indices = importance[dst_c].topk(k)
                dst_flat = dst_c * dst_h * dst_w + (dst_h // 2) * dst_w + (dst_w // 2)
                for src_c in top_indices.tolist():
                    src_flat = src_c * src_h * src_w + (src_h // 2) * src_w + (src_w // 2)
                    connections.append({
                        "src": src_flat,
                        "dst": dst_flat,
                        "weight": float(signed[dst_c, src_c]),
                    })
            result[name] = connections
    return result
