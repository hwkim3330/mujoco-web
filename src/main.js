
import * as THREE           from 'three';
import { GUI              } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls    } from 'three/addons/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import { setupGUI, downloadExampleScenesFolder, loadSceneFromURL, drawTendonsAndFlex, getPosition, getQuaternion, toMujocoPos, standardNormal } from './mujocoUtils.js';
import   load_mujoco        from 'https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js';
import { OnnxController   } from './onnxController.js';

// Load the MuJoCo Module
const mujoco = await load_mujoco();

// Set up Emscripten's Virtual File System
// Use humanoid.xml initially, then switch to OpenDuck after files are downloaded
var initialScene = "humanoid.xml";
mujoco.FS.mkdir('/working');
mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
mujoco.FS.writeFile("/working/" + initialScene, await(await fetch("./assets/scenes/" + initialScene)).text());

export class MuJoCoDemo {
  constructor() {
    this.mujoco = mujoco;

    // Load in the state from XML
    this.model = mujoco.MjModel.loadFromXML("/working/" + initialScene);
    this.data  = new mujoco.MjData(this.model);

    // Define Random State Variables
    this.params = { scene: initialScene, paused: false, help: false, ctrlnoiserate: 0.0, ctrlnoisestd: 0.0, keyframeNumber: 0, cameraFollow: true };
    this.mujoco_time = 0.0;
    this.bodies  = {}, this.lights = {};
    this.tmpVec  = new THREE.Vector3();
    this.tmpQuat = new THREE.Quaternion();
    this.updateGUICallbacks = [];

    // ONNX Controller
    this.onnxController = null;
    this.robotCommand = { x: 0, y: 0, rot: 0 };
    this.keysPressed = {};
    this.cameraOffset = new THREE.Vector3(0.5, 0.4, 0.5);
    this.dragForceScale = 2000;
    this.policyAutoPausedForDrag = false;

    this.container = document.createElement( 'div' );
    document.body.appendChild( this.container );

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.001, 100 );
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(2.0, 1.7, 1.7);
    this.scene.add(this.camera);

    this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
    this.scene.fog = new THREE.Fog(this.scene.background, 15, 25.5 );

    this.ambientLight = new THREE.AmbientLight( 0xffffff, 0.1 * 3.14 );
    this.ambientLight.name = 'AmbientLight';
    this.scene.add( this.ambientLight );

    this.spotlight = new THREE.SpotLight();
    this.spotlight.angle = 1.11;
    this.spotlight.distance = 10000;
    this.spotlight.penumbra = 0.5;
    this.spotlight.castShadow = true;
    this.spotlight.intensity = this.spotlight.intensity * 3.14 * 10.0;
    this.spotlight.shadow.mapSize.width = 1024;
    this.spotlight.shadow.mapSize.height = 1024;
    this.spotlight.shadow.camera.near = 0.1;
    this.spotlight.shadow.camera.far = 100;
    this.spotlight.position.set(0, 3, 3);
    const targetObject = new THREE.Object3D();
    this.scene.add(targetObject);
    this.spotlight.target = targetObject;
    targetObject.position.set(0, 1, 0);
    this.scene.add( this.spotlight );

    this.renderer = new THREE.WebGLRenderer( { antialias: true } );
    this.renderer.setPixelRatio(1.0);
    this.renderer.setSize( window.innerWidth, window.innerHeight );
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    THREE.ColorManagement.enabled = false;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.useLegacyLights = true;

    // Use manual RAF loop instead of setAnimationLoop (needed for async render)
    this.startAnimationLoop();

    this.container.appendChild( this.renderer.domElement );

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.7, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.10;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    window.addEventListener('resize', this.onWindowResize.bind(this));

    // Keyboard controls for robot
    this.setupKeyboardControls();

    // Initialize the Drag State Manager.
    this.dragStateManager = new DragStateManager(this.scene, this.renderer, this.camera, this.container.parentElement, this.controls);
  }

  startAnimationLoop() {
    const loop = async (timeMS) => {
      await this.render(timeMS);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  setupKeyboardControls() {
    // Key visual indicators
    const keyMap = {
      'KeyW': 'key-w', 'KeyA': 'key-a', 'KeyS': 'key-s', 'KeyD': 'key-d',
      'KeyQ': 'key-q', 'KeyE': 'key-e',
      'ArrowUp': 'key-w', 'ArrowDown': 'key-s', 'ArrowLeft': 'key-a', 'ArrowRight': 'key-d'
    };

    document.addEventListener('keydown', (e) => {
      this.keysPressed[e.code] = true;

      // Visual key feedback
      const keyEl = document.getElementById(keyMap[e.code]);
      if (keyEl) keyEl.classList.add('active');

      // P key: toggle ONNX policy
      if (e.code === 'KeyP') this.togglePolicy();

      // R key: reset robot to home keyframe
      if (e.code === 'KeyR') this.resetToHome();

      // Space: toggle pause
      if (e.code === 'Space') {
        this.params.paused = !this.params.paused;
        e.preventDefault();
      }

      this.updateRobotCommand();
    });
    document.addEventListener('keyup', (e) => {
      this.keysPressed[e.code] = false;
      const keyEl = document.getElementById(keyMap[e.code]);
      if (keyEl) keyEl.classList.remove('active');
      this.updateRobotCommand();
    });

    // Expose actions for on-screen buttons
    window._demoActions = {
      togglePolicy: () => this.togglePolicy(),
      resetPose: () => this.resetToHome(),
      togglePause: () => { this.params.paused = !this.params.paused; },
      headUp: () => {
        if (this.onnxController) {
          this.onnxController.commands[3] = Math.min(1.0, this.onnxController.commands[3] + 0.1);
        }
      },
      headDown: () => {
        if (this.onnxController) {
          this.onnxController.commands[3] = Math.max(0.2, this.onnxController.commands[3] - 0.1);
        }
      }
    };
  }

  togglePolicy() {
    if (!this.onnxController) return;
    this.onnxController.enabled = !this.onnxController.enabled;
    if (!this.onnxController.enabled) {
      this.onnxController.reset();
      console.log('ONNX policy DISABLED');
    } else {
      console.log('ONNX policy ENABLED');
    }
    this.updatePolicyUI();
  }

  updatePolicyUI() {
    const dot = document.getElementById('policy-dot');
    const label = document.getElementById('policy-label');
    const btn = document.getElementById('btn-policy');
    const btnMobile = document.getElementById('btn-policy-mobile');
    if (!this.onnxController) return;
    if (this.onnxController.enabled) {
      if (dot) { dot.className = 'dot green'; }
      if (label) label.textContent = 'ONNX';
      if (btn) { btn.textContent = 'P: Policy ON'; btn.className = 'action-btn active'; }
      if (btnMobile) { btnMobile.textContent = 'Policy ON'; btnMobile.className = 'action-btn active'; }
    } else {
      if (dot) { dot.className = 'dot yellow'; }
      if (label) label.textContent = 'OFF';
      if (btn) { btn.textContent = 'P: Policy OFF'; btn.className = 'action-btn warn'; }
      if (btnMobile) { btnMobile.textContent = 'Policy OFF'; btnMobile.className = 'action-btn warn'; }
    }
  }

  resetToHome() {
    if (this.model.nkey > 0) {
      const nq = this.model.nq;
      const nv = this.model.nv;
      this.data.qpos.set(this.model.key_qpos.slice(0, nq));
      for (let i = 0; i < nv; i++) this.data.qvel[i] = 0;
      if (this.model.key_ctrl) {
        this.data.ctrl.set(this.model.key_ctrl.slice(0, this.model.nu));
      }
      this.mujoco.mj_forward(this.model, this.data);
      // Warm up physics to settle contacts
      for (let i = 0; i < 100; i++) {
        this.mujoco.mj_step(this.model, this.data);
      }
      if (this.onnxController) {
        this.onnxController.reset();
        this.onnxController.enabled = true;
        this.updatePolicyUI();
      }
      // Reset timing to avoid burst of steps
      this.lastRenderTime = undefined;
      this.accumulator = 0;
      console.log('Reset to home keyframe (with warm-up)');
    }
  }

  updateRobotCommand() {
    // WASD / Arrow keys for movement
    // Default: slow forward walk
    let x = 0.06, y = 0, rot = 0;

    if (this.keysPressed['KeyW'] || this.keysPressed['ArrowUp']) x = 0.12;
    if (this.keysPressed['KeyS'] || this.keysPressed['ArrowDown']) x = -0.15;
    if (this.keysPressed['KeyA'] || this.keysPressed['ArrowLeft']) y += 0.15;
    if (this.keysPressed['KeyD'] || this.keysPressed['ArrowRight']) y -= 0.15;
    if (this.keysPressed['KeyQ']) rot += 0.5;
    if (this.keysPressed['KeyE']) rot -= 0.5;

    // 1/2 keys: adjust neck pitch (head up/down)
    if (this.onnxController) {
      if (this.keysPressed['Digit1']) {
        this.onnxController.commands[3] = Math.min(1.0, this.onnxController.commands[3] + 0.02);
      }
      if (this.keysPressed['Digit2']) {
        this.onnxController.commands[3] = Math.max(0.2, this.onnxController.commands[3] - 0.02);
      }
    }

    this.robotCommand = { x, y, rot };

    if (this.onnxController) {
      this.onnxController.setCommand(x, y, rot);
    }
  }

  async init() {
    // Download the examples to MuJoCo's virtual file system
    await downloadExampleScenesFolder(mujoco);

    // Load OpenDuck as default scene after files are ready
    const defaultScene = "openduck/scene_flat_terrain_backlash.xml";
    this.params.scene = defaultScene;

    // Initialize the three.js Scene using OpenDuck
    [this.model, this.data, this.bodies, this.lights] =
      await loadSceneFromURL(mujoco, defaultScene, this);

    // Apply home keyframe for OpenDuck
    if (this.model.nkey > 0) {
      const nq = this.model.nq;
      const nv = this.model.nv;
      const nu = this.model.nu;

      // Increase solver iterations for WASM accuracy
      // Training uses iterations=1 (fast for RL), but WASM needs more
      // iterations to converge to the same solution as native MuJoCo.
      // This prevents the head from drooping and causing backward fall.
      this.model.opt.iterations = 40;

      // Set positions from keyframe
      this.data.qpos.set(this.model.key_qpos.slice(0, nq));

      // Reset velocities to zero (critical for stable start!)
      for (let i = 0; i < nv; i++) {
        this.data.qvel[i] = 0;
      }

      // Set controls from keyframe
      if (this.model.key_ctrl) {
        this.data.ctrl.set(this.model.key_ctrl.slice(0, nu));
      }

      // Update forward kinematics
      mujoco.mj_forward(this.model, this.data);

      // Warm up physics: run 300 steps (~600ms sim time) to let contacts settle
      // and reach a physically consistent state before the policy takes over.
      for (let i = 0; i < 300; i++) {
        mujoco.mj_step(this.model, this.data);
      }
      console.log('Applied home keyframe + warm-up (300 steps, iterations=40)');
    }

    // Set camera for OpenDuck
    this.camera.position.set(0.5, 0.4, 0.5);
    this.controls.target.set(0, 0.15, 0);
    this.controls.update();

    // Initialize ONNX controller
    await this.initOnnxController();

    this.gui = new GUI();
    setupGUI(this);
  }

  async initOnnxController() {
    this.onnxController = new OnnxController(mujoco, this.model, this.data);

    try {
      const loaded = await this.onnxController.loadModel('./assets/models/openduck_walk.onnx');
      if (loaded) {
        this.onnxController.enabled = true;
        this.updatePolicyUI();
      }
    } catch (e) {
      console.warn('Failed to load ONNX model:', e);
      const dot = document.getElementById('policy-dot');
      const label = document.getElementById('policy-label');
      if (dot) dot.className = 'dot red';
      if (label) label.textContent = 'FAIL';
    }

    // Reset physics timing to avoid burst of steps on first frame
    this.lastRenderTime = undefined;
    this.accumulator = 0;

    // Hide loading screen
    const loadingScreen = document.getElementById('loading-screen');
    if (loadingScreen) loadingScreen.classList.add('hidden');
    setTimeout(() => { if (loadingScreen) loadingScreen.remove(); }, 600);

    // Listen for mobile joystick events
    window.addEventListener('joystick-move', (e) => {
      if (this.onnxController) {
        this.onnxController.setCommand(e.detail.x, e.detail.y, 0);
      }
    });
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize( window.innerWidth, window.innerHeight );
  }

  async render(timeMS) {
    this.controls.update();

    // Drag assist: pause policy while dragging so manual interaction wins.
    const dragActive = !!(this.dragStateManager && this.dragStateManager.active && this.dragStateManager.physicsObject);
    if (this.onnxController) {
      if (dragActive && this.onnxController.enabled && !this.policyAutoPausedForDrag) {
        this.onnxController.enabled = false;
        this.policyAutoPausedForDrag = true;
        this.updatePolicyUI();
      } else if (!dragActive && this.policyAutoPausedForDrag) {
        this.onnxController.enabled = true;
        this.policyAutoPausedForDrag = false;
        this.updatePolicyUI();
      }
    }

    if (!this.params["paused"]) {
      let timestep = this.model.opt.timestep;
      let timestepMs = timestep * 1000.0;

      // Fixed-rate physics: run exactly the right number of steps for this frame
      if (this.lastRenderTime === undefined) { this.lastRenderTime = timeMS; }
      let dt = timeMS - this.lastRenderTime;
      this.lastRenderTime = timeMS;

      // Cap dt to prevent spiral of death (max 50ms = 25 physics steps)
      if (dt > 50.0) dt = 50.0;

      // Accumulate time and step physics
      this.accumulator = (this.accumulator || 0) + dt;

      while (this.accumulator >= timestepMs) {

        // Jitter the control state with gaussian random noise (non-ONNX scenes)
        if (!(this.onnxController && this.onnxController.enabled) && this.params["ctrlnoisestd"] > 0.0) {
          let rate  = Math.exp(-timestep / Math.max(1e-10, this.params["ctrlnoiserate"]));
          let scale = this.params["ctrlnoisestd"] * Math.sqrt(1 - rate * rate);
          let currentCtrl = this.data.ctrl;
          for (let i = 0; i < currentCtrl.length; i++) {
            currentCtrl[i] = rate * currentCtrl[i] + scale * standardNormal();
            this.params["Actuator " + i] = currentCtrl[i];
          }
        }

        // Clear old perturbations, apply new ones.
        for (let i = 0; i < this.data.qfrc_applied.length; i++) { this.data.qfrc_applied[i] = 0.0; }
        let dragged = this.dragStateManager.physicsObject;
        if (dragged && dragged.bodyID) {
          for (let b = 0; b < this.model.nbody; b++) {
            if (this.bodies[b]) {
              getPosition  (this.data.xpos , b, this.bodies[b].position);
              getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
            }
          }
          // Propagate world matrices to child meshes so localToWorld is accurate
          dragged.updateWorldMatrix(true, false);
          let bodyID = dragged.bodyID;
          this.dragStateManager.update();
          let force = toMujocoPos(this.dragStateManager.currentWorld.clone().sub(this.dragStateManager.worldHit).multiplyScalar(this.model.body_mass[bodyID] * this.dragForceScale));
          let point = toMujocoPos(this.dragStateManager.worldHit.clone());
          mujoco.mj_applyFT(this.model, this.data, [force.x, force.y, force.z], [0, 0, 0], [point.x, point.y, point.z], bodyID, this.data.qfrc_applied);
        }

        // Physics step
        mujoco.mj_step(this.model, this.data);
        this.accumulator -= timestepMs;

        // Run policy synchronously at decimation boundary
        if (this.onnxController && this.onnxController.enabled && this.onnxController.session) {
          this.onnxController.stepCounter++;
          if (this.onnxController.stepCounter % this.onnxController.decimation === 0) {
            await this.onnxController.runPolicy();
          }
        }
      }

    } else if (this.params["paused"]) {
      this.dragStateManager.update();
      let dragged = this.dragStateManager.physicsObject;
      if (dragged && dragged.bodyID) {
        let b = dragged.bodyID;
        getPosition  (this.data.xpos , b, this.tmpVec , false);
        getQuaternion(this.data.xquat, b, this.tmpQuat, false);

        let offset = toMujocoPos(this.dragStateManager.currentWorld.clone()
          .sub(this.dragStateManager.worldHit).multiplyScalar(0.3));
        if (this.model.body_mocapid[b] >= 0) {
          console.log("Trying to move mocap body", b);
          let addr = this.model.body_mocapid[b] * 3;
          let pos  = this.data.mocap_pos;
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;
        } else {
          let root = this.model.body_rootid[b];
          let addr = this.model.jnt_qposadr[this.model.body_jntadr[root]];
          let pos  = this.data.qpos;
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;
        }
      }

      mujoco.mj_forward(this.model, this.data);
    }

    // Update body transforms.
    for (let b = 0; b < this.model.nbody; b++) {
      if (this.bodies[b]) {
        getPosition  (this.data.xpos , b, this.bodies[b].position);
        getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }

    // Update light transforms.
    for (let l = 0; l < this.model.nlight; l++) {
      if (this.lights[l]) {
        getPosition(this.data.light_xpos, l, this.lights[l].position);
        getPosition(this.data.light_xdir, l, this.tmpVec);
        this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
      }
    }

    // Update UI elements
    if (this.onnxController && this.params.scene.includes('openduck')) {
      const h = (this.data.qpos[2] || 0).toFixed(3);
      const contacts = this.onnxController.getFeetContacts();
      const elH = document.getElementById('stat-height');
      const elL = document.getElementById('stat-left');
      const elR = document.getElementById('stat-right');
      const elStep = document.getElementById('stat-step');
      const elSpeed = document.getElementById('speed-value');
      const elPause = document.getElementById('btn-pause');
      if (elH) elH.textContent = h;
      if (elL) { elL.textContent = contacts[0] ? 'Y' : 'N'; elL.style.color = contacts[0] ? '#4caf50' : '#666'; }
      if (elR) { elR.textContent = contacts[1] ? 'Y' : 'N'; elR.style.color = contacts[1] ? '#4caf50' : '#666'; }
      const elNeck = document.getElementById('stat-neck');
      if (elStep) elStep.textContent = this.onnxController.policyStepCount;
      if (elSpeed) elSpeed.textContent = Math.abs(this.onnxController.commands[0]).toFixed(2);
      if (elNeck) elNeck.textContent = this.onnxController.commands[3].toFixed(1);
      if (elPause) {
        if (this.params.paused) {
          elPause.textContent = 'Space: PAUSED';
          elPause.className = 'action-btn warn';
        } else {
          elPause.textContent = 'Space: Pause';
          elPause.className = 'action-btn';
        }
      }
    }

    // Camera follow - track the duck's base body
    if (this.params.cameraFollow && this.params.scene.includes('openduck')) {
      const baseX = this.data.qpos[0];
      const baseY = this.data.qpos[1];
      const baseZ = this.data.qpos[2];
      // Update orbit controls target to follow duck (swizzle Y/Z for three.js)
      this.controls.target.set(baseX, baseZ, -baseY);
    }

    // Draw Tendons and Flex verts
    drawTendonsAndFlex(this.mujocoRoot, this.model, this.data);

    // Render!
    this.renderer.render( this.scene, this.camera );
  }
}

let demo = new MuJoCoDemo();
await demo.init();
