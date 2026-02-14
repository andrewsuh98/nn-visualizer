import random

import torch
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
    with torch.no_grad():
        flat = x_normalized.view(1, -1)
        fc1_out = model.relu(model.fc1(flat))
        fc2_out = model.relu(model.fc2(fc1_out))
        logits = model.fc3(fc2_out)
        probs = torch.softmax(logits, dim=1)
    return {
        "fc1_relu": fc1_out.squeeze().tolist(),
        "fc2_relu": fc2_out.squeeze().tolist(),
        "logits": logits.squeeze().tolist(),
        "probabilities": probs.squeeze().tolist(),
    }


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
    for name, dst_size in [
        ("fc1", 128),
        ("fc2", 64),
        ("fc3", 10),
    ]:
        layer = getattr(model, name)
        w = layer.weight.data  # shape: (dst_size, src_size)
        connections = []
        for dst in range(dst_size):
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
