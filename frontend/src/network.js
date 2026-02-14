import * as THREE from "three";
import { NEURON_RADIUS, NEURON_SPACING } from "./constants.js";

const INACTIVE_COLOR = new THREE.Color(0x222233);

// Compute the 3D position for neuron `idx` within a layer
function neuronPosition(layer, idx) {
  const col = idx % layer.cols;
  const row = Math.floor(idx / layer.cols);
  const x = (col - (layer.cols - 1) / 2) * NEURON_SPACING;
  const y = ((layer.rows - 1) / 2 - row) * NEURON_SPACING;
  return new THREE.Vector3(x, y, layer.z);
}

export function buildNeurons(scene, layers) {
  const geometry = new THREE.SphereGeometry(NEURON_RADIUS, 12, 8);
  const layerMeshes = {};

  for (const layer of layers) {
    const material = new THREE.MeshPhongMaterial({ color: 0xffffff });
    const mesh = new THREE.InstancedMesh(geometry, material, layer.size);

    const dummy = new THREE.Object3D();

    for (let i = 0; i < layer.size; i++) {
      const pos = neuronPosition(layer, i);
      dummy.position.copy(pos);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, INACTIVE_COLOR);
    }

    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
    scene.add(mesh);
    layerMeshes[layer.name] = mesh;
  }

  return layerMeshes;
}

export function buildConnections(scene, weightsData, layers, weightLayers) {
  const connectionMeshes = {};

  for (let i = 0; i < weightLayers.length; i++) {
    const weightName = weightLayers[i];
    const srcLayer = layers[i];
    const dstLayer = layers[i + 1];
    const connections = weightsData[weightName];

    const positions = [];
    const colors = [];

    for (const conn of connections) {
      const srcPos = neuronPosition(srcLayer, conn.src);
      const dstPos = neuronPosition(dstLayer, conn.dst);

      positions.push(srcPos.x, srcPos.y, srcPos.z);
      positions.push(dstPos.x, dstPos.y, dstPos.z);

      const w = conn.weight;
      const magnitude = Math.min(Math.abs(w) / 2, 1);
      const c = w > 0
        ? new THREE.Color(0.2, 0.3, 1).multiplyScalar(0.3 + magnitude * 0.7)
        : new THREE.Color(1, 0.2, 0.2).multiplyScalar(0.3 + magnitude * 0.7);

      colors.push(c.r, c.g, c.b);
      colors.push(c.r, c.g, c.b);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });

    const lines = new THREE.LineSegments(geometry, material);
    scene.add(lines);
    connectionMeshes[weightName] = lines;
  }

  return connectionMeshes;
}

export function setLayerColors(layerMeshes, layerName, colorArray) {
  const mesh = layerMeshes[layerName];
  const color = new THREE.Color();
  for (let i = 0; i < colorArray.length; i++) {
    const [r, g, b] = colorArray[i];
    color.setRGB(r, g, b);
    mesh.setColorAt(i, color);
  }
  mesh.instanceColor.needsUpdate = true;
}

export function resetAllLayers(layerMeshes, layers) {
  for (const layer of layers) {
    const mesh = layerMeshes[layer.name];
    for (let i = 0; i < layer.size; i++) {
      mesh.setColorAt(i, INACTIVE_COLOR);
    }
    mesh.instanceColor.needsUpdate = true;
  }
}

export function resetConnections(connectionMeshes, weightLayers) {
  for (const name of weightLayers) {
    connectionMeshes[name].material.opacity = 0;
  }
}

export { neuronPosition };
