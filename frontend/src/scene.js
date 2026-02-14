import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { BG_COLOR } from "./constants.js";

export function createScene(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG_COLOR);

  const camera = new THREE.PerspectiveCamera(
    60,
    container.clientWidth / container.clientHeight,
    0.1,
    500
  );
  camera.position.set(-70, 25, 110);
  camera.lookAt(0, 0, 45);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 45);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.update();

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(20, 40, 60);
  scene.add(dirLight);

  window.addEventListener("resize", () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  return { scene, camera, renderer, controls };
}
