/**
 * ONNX-based Robot Controller for OpenDuck Mini
 * Ported from Open_Duck_Playground/playground/open_duck_mini_v2/mujoco_infer.py
 */

export class OnnxController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.session = null;
    this.enabled = false;

    // Params from Playground config
    this.actionScale = 0.25;
    this.dofVelScale = 0.05;
    this.maxMotorVelocity = 5.24; // rad/s
    this.simDt = 0.002;
    this.decimation = 10;
    this.ctrlDt = this.simDt * this.decimation; // 0.02

    // Number of actuators (10 for legs only, 14 for full model)
    this.numDofs = this.model.nu;

    // State
    this.lastAction = null;
    this.lastLastAction = null;
    this.lastLastLastAction = null;
    this.motorTargets = null;
    this.prevMotorTargets = null;
    this.defaultActuator = null;

    // Commands [lin_vel_x, lin_vel_y, ang_vel, neck_pitch, head_pitch, head_yaw, head_roll]
    this.commands = [0, 0, 0, 0, 0, 0, 0];

    // Imitation phase
    this.imitationI = 0;
    this.nbStepsInPeriod = 50; // From PolyReferenceMotion
    this.imitationPhase = [0, 0];

    // For async inference
    this.pendingAction = null;
    this.inferenceRunning = false;
    this.stepCounter = 0;

    // Sensor addresses (will be set after model loads)
    this.gyroAddr = -1;
    this.accelAddr = -1;
  }

  async loadModel(url) {
    try {
      if (typeof ort === 'undefined') {
        console.warn('ONNX Runtime not loaded.');
        return false;
      }

      this.session = await ort.InferenceSession.create(url);
      console.log('ONNX model loaded:', url);
      console.log('Input names:', this.session.inputNames);
      console.log('Output names:', this.session.outputNames);

      this.inputName = this.session.inputNames[0];
      this.outputName = this.session.outputNames[0];

      this.initState();
      this.findSensorAddresses();

      return true;
    } catch (e) {
      console.error('Failed to load ONNX model:', e);
      return false;
    }
  }

  initState() {
    const n = this.numDofs;
    this.lastAction = new Float32Array(n);
    this.lastLastAction = new Float32Array(n);
    this.lastLastLastAction = new Float32Array(n);
    this.motorTargets = new Float32Array(n);
    this.prevMotorTargets = new Float32Array(n);
    this.defaultActuator = new Float32Array(n);

    // Get default actuator positions from keyframe "home"
    // These values are from scene_flat_terrain.xml keyframe
    const homeCtrl = [
      0.002,   // left_hip_yaw
      0.053,   // left_hip_roll
      -0.63,   // left_hip_pitch
      1.368,   // left_knee
      -0.784,  // left_ankle
      0,       // neck_pitch
      0,       // head_pitch
      0,       // head_yaw
      0,       // head_roll
      -0.003,  // right_hip_yaw
      -0.065,  // right_hip_roll
      0.635,   // right_hip_pitch
      1.379,   // right_knee
      -0.796   // right_ankle
    ];

    for (let i = 0; i < n; i++) {
      this.defaultActuator[i] = homeCtrl[i] || 0;
      this.motorTargets[i] = this.defaultActuator[i];
      this.prevMotorTargets[i] = this.defaultActuator[i];
    }

    console.log(`Initialized with ${n} actuators, defaultActuator:`, Array.from(this.defaultActuator));
  }

  findSensorAddresses() {
    // Find gyro and accelerometer sensor addresses
    // In MuJoCo, sensors are stored in sensordata array
    const nsensor = this.model.nsensor;

    for (let i = 0; i < nsensor; i++) {
      const adr = this.model.sensor_adr[i];
      const dim = this.model.sensor_dim[i];
      // Gyro is typically sensor type 8, accelerometer is type 9
      const type = this.model.sensor_type[i];

      if (type === 8 && this.gyroAddr < 0) { // mjSENS_GYRO
        this.gyroAddr = adr;
        console.log('Found gyro at address:', adr);
      }
      if (type === 9 && this.accelAddr < 0) { // mjSENS_ACCELEROMETER
        this.accelAddr = adr;
        console.log('Found accelerometer at address:', adr);
      }
    }
  }

  setCommand(linX, linY, angZ) {
    // Clamp to valid ranges from Playground config
    this.commands[0] = Math.max(-0.15, Math.min(0.15, linX));
    this.commands[1] = Math.max(-0.2, Math.min(0.2, linY));
    this.commands[2] = Math.max(-1.0, Math.min(1.0, angZ));
    // Head commands (3-6) left at 0 for now
  }

  getGyro() {
    if (this.gyroAddr >= 0) {
      return [
        this.data.sensordata[this.gyroAddr],
        this.data.sensordata[this.gyroAddr + 1],
        this.data.sensordata[this.gyroAddr + 2]
      ];
    }
    return [0, 0, 0];
  }

  getAccelerometer() {
    if (this.accelAddr >= 0) {
      const ax = this.data.sensordata[this.accelAddr] + 1.3; // Gravity compensation
      const ay = this.data.sensordata[this.accelAddr + 1];
      const az = this.data.sensordata[this.accelAddr + 2];
      return [ax, ay, az];
    }
    return [1.3, 0, 0]; // Default with gravity compensation
  }

  getActuatorJointsQpos() {
    // Get joint positions for actuated joints
    // Skip first 7 (floating base: 3 pos + 4 quat)
    const angles = new Float32Array(this.numDofs);
    const qpos = this.data.qpos;
    for (let i = 0; i < this.numDofs; i++) {
      angles[i] = qpos[7 + i] || 0;
    }
    return angles;
  }

  getActuatorJointsQvel() {
    // Get joint velocities for actuated joints
    // Skip first 6 (floating base: 3 linear + 3 angular vel)
    const vels = new Float32Array(this.numDofs);
    const qvel = this.data.qvel;
    for (let i = 0; i < this.numDofs; i++) {
      vels[i] = qvel[6 + i] || 0;
    }
    return vels;
  }

  getFeetContacts() {
    // Simple height-based contact detection
    // TODO: Implement proper geom collision check
    const height = this.data.qpos[2] || 0;
    if (height < 0.18) {
      return [1.0, 1.0];
    }
    return [0.0, 0.0];
  }

  getObservation() {
    // Build observation exactly as in mujoco_infer.py get_obs()
    const obs = [];

    // Gyro (3)
    const gyro = this.getGyro();
    obs.push(...gyro);

    // Accelerometer (3) with gravity compensation
    const accel = this.getAccelerometer();
    obs.push(...accel);

    // Commands (7)
    obs.push(...this.commands);

    // Joint angles - default (numDofs)
    const jointAngles = this.getActuatorJointsQpos();
    for (let i = 0; i < this.numDofs; i++) {
      obs.push(jointAngles[i] - this.defaultActuator[i]);
    }

    // Joint velocities * dof_vel_scale (numDofs)
    const jointVel = this.getActuatorJointsQvel();
    for (let i = 0; i < this.numDofs; i++) {
      obs.push(jointVel[i] * this.dofVelScale);
    }

    // Last action (numDofs)
    obs.push(...this.lastAction);

    // Last last action (numDofs)
    obs.push(...this.lastLastAction);

    // Last last last action (numDofs)
    obs.push(...this.lastLastLastAction);

    // Motor targets (numDofs)
    obs.push(...this.motorTargets);

    // Foot contacts (2)
    const contacts = this.getFeetContacts();
    obs.push(...contacts);

    // Imitation phase (2)
    obs.push(...this.imitationPhase);

    return new Float32Array(obs);
  }

  step(timestep) {
    if (!this.session || !this.enabled) {
      return;
    }

    this.stepCounter++;

    // Only run policy at control frequency (every decimation steps)
    if (this.stepCounter % this.decimation !== 0) {
      // Still apply motor targets
      if (this.motorTargets) {
        const ctrl = this.data.ctrl;
        for (let i = 0; i < Math.min(this.numDofs, ctrl.length); i++) {
          ctrl[i] = this.motorTargets[i];
        }
      }
      return;
    }

    // Update imitation phase
    this.imitationI = (this.imitationI + 1) % this.nbStepsInPeriod;
    const phase = (this.imitationI / this.nbStepsInPeriod) * 2 * Math.PI;
    this.imitationPhase[0] = Math.cos(phase);
    this.imitationPhase[1] = Math.sin(phase);

    // Apply pending action from previous inference
    if (this.pendingAction) {
      // Update action history
      this.lastLastLastAction.set(this.lastLastAction);
      this.lastLastAction.set(this.lastAction);
      this.lastAction.set(this.pendingAction);

      // Compute motor targets
      for (let i = 0; i < this.numDofs; i++) {
        this.motorTargets[i] = this.defaultActuator[i] + this.pendingAction[i] * this.actionScale;
      }

      // Apply velocity limits
      for (let i = 0; i < this.numDofs; i++) {
        const maxChange = this.maxMotorVelocity * this.ctrlDt;
        const diff = this.motorTargets[i] - this.prevMotorTargets[i];
        if (Math.abs(diff) > maxChange) {
          this.motorTargets[i] = this.prevMotorTargets[i] + Math.sign(diff) * maxChange;
        }
        this.prevMotorTargets[i] = this.motorTargets[i];
      }

      // Apply to MuJoCo
      const ctrl = this.data.ctrl;
      for (let i = 0; i < Math.min(this.numDofs, ctrl.length); i++) {
        ctrl[i] = this.motorTargets[i];
      }

      this.pendingAction = null;
    }

    // Start new inference
    if (!this.inferenceRunning) {
      this.runInference();
    }
  }

  async runInference() {
    if (this.inferenceRunning) return;
    this.inferenceRunning = true;

    try {
      const obs = this.getObservation();

      // Debug: log observation size and sample values
      if (this.stepCounter <= 20) {
        console.log(`Step ${this.stepCounter}: obs size=${obs.length}, gyro=[${obs[0].toFixed(3)}, ${obs[1].toFixed(3)}, ${obs[2].toFixed(3)}]`);
      }

      // Run ONNX inference
      const inputTensor = new ort.Tensor('float32', obs, [1, obs.length]);
      const feeds = {};
      feeds[this.inputName] = inputTensor;

      const results = await this.session.run(feeds);
      const output = results[this.outputName];

      if (output) {
        this.pendingAction = new Float32Array(output.data);
        // Debug: log action
        if (this.stepCounter <= 20) {
          console.log(`Action: [${this.pendingAction.slice(0, 5).map(v => v.toFixed(3)).join(', ')}...]`);
        }
      }
    } catch (e) {
      console.error('ONNX inference error:', e);
    }

    this.inferenceRunning = false;
  }

  reset() {
    if (this.lastAction) this.lastAction.fill(0);
    if (this.lastLastAction) this.lastLastAction.fill(0);
    if (this.lastLastLastAction) this.lastLastLastAction.fill(0);

    this.imitationI = 0;
    this.imitationPhase = [0, 0];
    this.commands = [0, 0, 0, 0, 0, 0, 0];
    this.stepCounter = 0;
    this.pendingAction = null;

    if (this.motorTargets && this.defaultActuator) {
      this.motorTargets.set(this.defaultActuator);
      this.prevMotorTargets.set(this.defaultActuator);
    }
  }
}
