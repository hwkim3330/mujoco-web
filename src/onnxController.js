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

    // Params from Playground config (must match training)
    this.actionScale = 0.25;
    this.dofVelScale = 0.05;
    this.maxMotorVelocity = 5.24; // rad/s
    this.simDt = 0.002;
    this.decimation = 10;
    this.ctrlDt = this.simDt * this.decimation; // 0.02

    // Number of actuators (14 for full model with head)
    this.numDofs = 14;

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
    this.nbStepsInPeriod = 50;
    this.imitationPhase = [0, 0];

    // Current action (computed synchronously)
    this.currentAction = null;
    this.stepCounter = 0;

    // Sensor addresses
    this.gyroAddr = -1;
    this.accelAddr = -1;

    // Contact geom names for foot detection
    this.leftFootGeomId = -1;
    this.rightFootGeomId = -1;
    this.floorGeomId = -1;
  }

  async loadModel(url) {
    try {
      if (typeof ort === 'undefined') {
        console.warn('ONNX Runtime not loaded.');
        return false;
      }

      this.session = await ort.InferenceSession.create(url);
      console.log('ONNX model loaded:', url);
      console.log('Input:', this.session.inputNames[0], 'Output:', this.session.outputNames[0]);

      this.inputName = this.session.inputNames[0];
      this.outputName = this.session.outputNames[0];

      this.numDofs = this.model.nu;
      this.initState();
      this.findSensorAddresses();
      this.findContactGeoms();

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
    this.currentAction = null;

    // Get default actuator positions from model keyframe
    if (this.model.nkey > 0 && this.model.key_ctrl) {
      for (let i = 0; i < n; i++) {
        this.defaultActuator[i] = this.model.key_ctrl[i] || 0;
      }
    } else {
      // Fallback: hardcoded values from scene_flat_terrain.xml keyframe
      const homeCtrl = [
        0.002, 0.053, -0.63, 1.368, -0.784,
        0, 0, 0, 0,
        -0.003, -0.065, 0.635, 1.379, -0.796
      ];
      for (let i = 0; i < n; i++) {
        this.defaultActuator[i] = homeCtrl[i] || 0;
      }
    }

    for (let i = 0; i < n; i++) {
      this.motorTargets[i] = this.defaultActuator[i];
      this.prevMotorTargets[i] = this.defaultActuator[i];
    }
  }

  findSensorAddresses() {
    const nsensor = this.model.nsensor;
    const names = this.model.names;

    const getNameAt = (nameAdr) => {
      if (!names || nameAdr < 0 || nameAdr >= names.length) return '';
      let name = '';
      for (let j = nameAdr; j < names.length && names[j] !== 0; j++) {
        name += String.fromCharCode(names[j]);
      }
      return name;
    };

    // Search by name
    if (names && this.model.name_sensoradr) {
      for (let i = 0; i < nsensor; i++) {
        const nameAdr = this.model.name_sensoradr[i];
        const sensorName = getNameAt(nameAdr);
        const adr = this.model.sensor_adr[i];

        if (sensorName === 'gyro') {
          this.gyroAddr = adr;
        }
        if (sensorName === 'accelerometer') {
          this.accelAddr = adr;
        }
      }
    }

    // Fallback by sensor type (mjSENS_GYRO = 8, mjSENS_ACCELEROMETER = 9)
    if (this.gyroAddr < 0 || this.accelAddr < 0) {
      for (let i = 0; i < nsensor; i++) {
        const type = this.model.sensor_type[i];
        const adr = this.model.sensor_adr[i];
        if (type === 8 && this.gyroAddr < 0) this.gyroAddr = adr;
        if (type === 9 && this.accelAddr < 0) this.accelAddr = adr;
      }
    }

    // Hardcoded fallback
    if (this.gyroAddr < 0) this.gyroAddr = 0;
    if (this.accelAddr < 0) this.accelAddr = 6;
  }

  findContactGeoms() {
    // Find geom IDs for foot contact detection
    const names = this.model.names;
    const getNameAt = (nameAdr) => {
      if (!names || nameAdr < 0 || nameAdr >= names.length) return '';
      let name = '';
      for (let j = nameAdr; j < names.length && names[j] !== 0; j++) {
        name += String.fromCharCode(names[j]);
      }
      return name;
    };

    for (let g = 0; g < this.model.ngeom; g++) {
      if (this.model.name_geomadr) {
        const nameAdr = this.model.name_geomadr[g];
        const geomName = getNameAt(nameAdr);
        if (geomName === 'left_foot_bottom_tpu') this.leftFootGeomId = g;
        if (geomName === 'right_foot_bottom_tpu') this.rightFootGeomId = g;
        if (geomName === 'floor') this.floorGeomId = g;
      }
    }
  }

  setCommand(linX, linY, angZ) {
    this.commands[0] = Math.max(-0.15, Math.min(0.15, linX));
    this.commands[1] = Math.max(-0.2, Math.min(0.2, linY));
    this.commands[2] = Math.max(-1.0, Math.min(1.0, angZ));
  }

  getGyro() {
    return [
      this.data.sensordata[this.gyroAddr],
      this.data.sensordata[this.gyroAddr + 1],
      this.data.sensordata[this.gyroAddr + 2]
    ];
  }

  getAccelerometer() {
    const ax = this.data.sensordata[this.accelAddr] + 1.3; // Gravity compensation (matches training)
    const ay = this.data.sensordata[this.accelAddr + 1];
    const az = this.data.sensordata[this.accelAddr + 2];
    return [ax, ay, az];
  }

  getActuatorJointsQpos() {
    // Skip first 7 (floating base: 3 pos + 4 quat)
    const angles = new Float32Array(this.numDofs);
    const qpos = this.data.qpos;
    for (let i = 0; i < this.numDofs; i++) {
      angles[i] = qpos[7 + i] || 0;
    }
    return angles;
  }

  getActuatorJointsQvel() {
    // Skip first 6 (floating base: 3 linear + 3 angular vel)
    const vels = new Float32Array(this.numDofs);
    const qvel = this.data.qvel;
    for (let i = 0; i < this.numDofs; i++) {
      vels[i] = qvel[6 + i] || 0;
    }
    return vels;
  }

  getFeetContacts() {
    // Check MuJoCo contact data for foot-floor contacts
    let leftContact = 0.0;
    let rightContact = 0.0;

    const ncon = this.data.ncon;
    for (let c = 0; c < ncon; c++) {
      // Each contact has geom1 and geom2
      const geom1 = this.data.contact_geom1 ? this.data.contact_geom1[c] : -1;
      const geom2 = this.data.contact_geom2 ? this.data.contact_geom2[c] : -1;

      // Check if left foot contacts floor
      if ((geom1 === this.leftFootGeomId && geom2 === this.floorGeomId) ||
          (geom2 === this.leftFootGeomId && geom1 === this.floorGeomId)) {
        leftContact = 1.0;
      }
      // Check if right foot contacts floor
      if ((geom1 === this.rightFootGeomId && geom2 === this.floorGeomId) ||
          (geom2 === this.rightFootGeomId && geom1 === this.floorGeomId)) {
        rightContact = 1.0;
      }
    }

    // Fallback: if contact data arrays not available, use body height heuristic
    if (!this.data.contact_geom1 || !this.data.contact_geom2) {
      // Use foot site positions from sensordata if available
      // Otherwise simple height-based detection
      const height = this.data.qpos[2] || 0;
      if (height < 0.18) {
        leftContact = 1.0;
        rightContact = 1.0;
      }
    }

    return [leftContact, rightContact];
  }

  getObservation() {
    // Build observation exactly as in mujoco_infer.py get_obs()
    const obs = [];

    // Gyro (3)
    obs.push(...this.getGyro());

    // Accelerometer (3) with gravity compensation
    obs.push(...this.getAccelerometer());

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
    obs.push(...this.getFeetContacts());

    // Imitation phase (2)
    obs.push(...this.imitationPhase);

    return new Float32Array(obs);
  }

  /**
   * Synchronous step - called every physics step from the render loop.
   * At every decimation boundary, runs inference synchronously using cached action.
   */
  step(timestep) {
    if (!this.session || !this.enabled) return;

    this.stepCounter++;

    // Apply motor targets every step
    const ctrl = this.data.ctrl;
    for (let i = 0; i < Math.min(this.numDofs, ctrl.length); i++) {
      ctrl[i] = this.motorTargets[i];
    }

    // Only compute new action at control frequency (every decimation steps)
    if (this.stepCounter % this.decimation !== 0) return;

    // Update imitation phase
    this.imitationI = (this.imitationI + 1) % this.nbStepsInPeriod;
    const phase = (this.imitationI / this.nbStepsInPeriod) * 2 * Math.PI;
    this.imitationPhase[0] = Math.cos(phase);
    this.imitationPhase[1] = Math.sin(phase);

    // Apply current action (computed at previous control step or from initial inference)
    if (this.currentAction) {
      // Update action history
      this.lastLastLastAction.set(this.lastLastAction);
      this.lastLastAction.set(this.lastAction);
      this.lastAction.set(this.currentAction);

      // Compute motor targets: default + action * scale
      for (let i = 0; i < this.numDofs; i++) {
        this.motorTargets[i] = this.defaultActuator[i] + this.currentAction[i] * this.actionScale;
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

      // Apply to ctrl
      for (let i = 0; i < Math.min(this.numDofs, ctrl.length); i++) {
        ctrl[i] = this.motorTargets[i];
      }
    }

    // Start async inference for next control step
    this.runInferenceAsync();
  }

  async runInferenceAsync() {
    try {
      const obs = this.getObservation();

      // Check for NaN
      if (obs.some(v => isNaN(v))) {
        console.warn('Observation contains NaN, skipping inference');
        return;
      }

      const inputTensor = new ort.Tensor('float32', obs, [1, obs.length]);
      const feeds = {};
      feeds[this.inputName] = inputTensor;

      const results = await this.session.run(feeds);
      const output = results[this.outputName];

      if (output) {
        this.currentAction = new Float32Array(output.data);
      }
    } catch (e) {
      console.error('ONNX inference error:', e);
    }
  }

  /**
   * Run first inference to have an action ready before simulation starts.
   */
  async runFirstInference() {
    try {
      const obs = this.getObservation();
      const inputTensor = new ort.Tensor('float32', obs, [1, obs.length]);
      const feeds = {};
      feeds[this.inputName] = inputTensor;

      const results = await this.session.run(feeds);
      const output = results[this.outputName];

      if (output) {
        this.currentAction = new Float32Array(output.data);
      }
    } catch (e) {
      console.error('Initial inference error:', e);
    }
  }

  reset() {
    if (this.lastAction) this.lastAction.fill(0);
    if (this.lastLastAction) this.lastLastAction.fill(0);
    if (this.lastLastLastAction) this.lastLastLastAction.fill(0);

    this.imitationI = 0;
    this.imitationPhase = [0, 0];
    this.commands = [0, 0, 0, 0, 0, 0, 0];
    this.stepCounter = 0;
    this.currentAction = null;

    if (this.motorTargets && this.defaultActuator) {
      this.motorTargets.set(this.defaultActuator);
      this.prevMotorTargets.set(this.defaultActuator);
    }
  }
}
