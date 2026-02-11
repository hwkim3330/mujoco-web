/**
 * ONNX-based Robot Controller for OpenDuck Mini
 * Ported from Open_Duck_Playground/playground/open_duck_mini_v2/mujoco_infer.py
 *
 * Control loop matches Python EXACTLY:
 *   1. mj_step (physics)
 *   2. counter++
 *   3. if counter % decimation == 0:
 *        a. update imitation phase
 *        b. obs = get_obs()        (uses OLD action history, OLD motor_targets)
 *        c. action = policy(obs)    (synchronous via await)
 *        d. update action history   (AFTER obs)
 *        e. motor_targets = default + action * scale
 *        f. velocity clamp
 *        g. data.ctrl = motor_targets
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

    // Imitation phase (updated before first obs, so initial value doesn't matter)
    this.imitationI = 0;
    this.nbStepsInPeriod = 50;
    this.imitationPhase = [0, 0]; // Matches Python init; updated before first use

    // Step counter - incremented AFTER each mj_step (matches Python)
    this.stepCounter = 0;

    // Sensor addresses
    this.gyroAddr = -1;
    this.accelAddr = -1;
    this.leftFootPosAddr = -1;
    this.rightFootPosAddr = -1;
  }

  async loadModel(url) {
    try {
      if (typeof ort === 'undefined') {
        console.warn('ONNX Runtime not loaded.');
        return false;
      }

      this.session = await ort.InferenceSession.create(url);
      console.log('ONNX model loaded:', url);

      this.inputName = this.session.inputNames[0];
      this.outputName = this.session.outputNames[0];

      this.numDofs = this.model.nu;
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

    if (this.model.nkey > 0 && this.model.key_ctrl) {
      for (let i = 0; i < n; i++) {
        this.defaultActuator[i] = this.model.key_ctrl[i] || 0;
      }
    } else {
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

    // Set initial ctrl to default (matches Python: data.ctrl[:] = default_actuator)
    const ctrl = this.data.ctrl;
    for (let i = 0; i < Math.min(n, ctrl.length); i++) {
      ctrl[i] = this.motorTargets[i];
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

    if (names && this.model.name_sensoradr) {
      for (let i = 0; i < nsensor; i++) {
        const nameAdr = this.model.name_sensoradr[i];
        const sensorName = getNameAt(nameAdr);
        const adr = this.model.sensor_adr[i];

        if (sensorName === 'gyro') this.gyroAddr = adr;
        if (sensorName === 'accelerometer') this.accelAddr = adr;
        if (sensorName === 'left_foot_pos') this.leftFootPosAddr = adr;
        if (sensorName === 'right_foot_pos') this.rightFootPosAddr = adr;
      }
    }

    // Fallback by sensor type
    if (this.gyroAddr < 0 || this.accelAddr < 0) {
      for (let i = 0; i < nsensor; i++) {
        const type = this.model.sensor_type[i];
        const adr = this.model.sensor_adr[i];
        if (type === 8 && this.gyroAddr < 0) this.gyroAddr = adr;
        if (type === 9 && this.accelAddr < 0) this.accelAddr = adr;
      }
    }

    if (this.gyroAddr < 0) this.gyroAddr = 0;
    if (this.accelAddr < 0) this.accelAddr = 6;

    console.log(`Sensors: gyro@${this.gyroAddr}, accel@${this.accelAddr}, leftFoot@${this.leftFootPosAddr}, rightFoot@${this.rightFootPosAddr}`);
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
    // Match Python: accelerometer[0] += 1.3
    return [
      this.data.sensordata[this.accelAddr] + 1.3,
      this.data.sensordata[this.accelAddr + 1],
      this.data.sensordata[this.accelAddr + 2]
    ];
  }

  getActuatorJointsQpos() {
    const angles = new Float32Array(this.numDofs);
    for (let i = 0; i < this.numDofs; i++) {
      angles[i] = this.data.qpos[7 + i] || 0;
    }
    return angles;
  }

  getActuatorJointsQvel() {
    const vels = new Float32Array(this.numDofs);
    for (let i = 0; i < this.numDofs; i++) {
      vels[i] = this.data.qvel[6 + i] || 0;
    }
    return vels;
  }

  getFeetContacts() {
    // Use foot position sensors for WASM contact detection
    // Check if foot z-position is near ground (z close to 0)
    const contactThreshold = 0.025; // meters (generous to avoid false negatives)

    if (this.leftFootPosAddr >= 0 && this.rightFootPosAddr >= 0) {
      const leftZ = this.data.sensordata[this.leftFootPosAddr + 2];
      const rightZ = this.data.sensordata[this.rightFootPosAddr + 2];
      return [
        leftZ < contactThreshold ? 1.0 : 0.0,
        rightZ < contactThreshold ? 1.0 : 0.0
      ];
    }

    // Fallback: base height heuristic
    const height = this.data.qpos[2] || 0;
    return height < 0.18 ? [1.0, 1.0] : [0.0, 0.0];
  }

  getObservation() {
    const obs = [];

    // Gyro (3)
    obs.push(...this.getGyro());

    // Accelerometer (3) with bias
    obs.push(...this.getAccelerometer());

    // Commands (7)
    obs.push(...this.commands);

    // Joint angles - default (numDofs)
    const jointAngles = this.getActuatorJointsQpos();
    for (let i = 0; i < this.numDofs; i++) {
      obs.push(jointAngles[i] - this.defaultActuator[i]);
    }

    // Joint velocities * scale (numDofs)
    const jointVel = this.getActuatorJointsQvel();
    for (let i = 0; i < this.numDofs; i++) {
      obs.push(jointVel[i] * this.dofVelScale);
    }

    // Last actions (numDofs * 3) - MUST use values from BEFORE this step's update
    obs.push(...this.lastAction);
    obs.push(...this.lastLastAction);
    obs.push(...this.lastLastLastAction);

    // Motor targets (numDofs) - MUST use values from BEFORE this step's update
    obs.push(...this.motorTargets);

    // Foot contacts (2)
    obs.push(...this.getFeetContacts());

    // Imitation phase (2)
    obs.push(...this.imitationPhase);

    return new Float32Array(obs);
  }

  /**
   * Run policy at decimation boundary. Called AFTER mj_step.
   * Matches Python mujoco_infer.py exactly:
   *   1. Update imitation phase
   *   2. Build observation (OLD action history, OLD motor_targets)
   *   3. Run ONNX inference (synchronous via await)
   *   4. Update action history (AFTER obs and inference)
   *   5. Compute new motor_targets
   *   6. Velocity clamp
   *   7. Apply to ctrl (persists for next decimation physics steps)
   */
  async runPolicy() {
    if (!this.session || !this.enabled) return;

    // 1. Update imitation phase (matches Python: increment BEFORE obs)
    this.imitationI = (this.imitationI + 1) % this.nbStepsInPeriod;
    const phase = (this.imitationI / this.nbStepsInPeriod) * 2 * Math.PI;
    this.imitationPhase[0] = Math.cos(phase);
    this.imitationPhase[1] = Math.sin(phase);

    // 2. Build observation with CURRENT (old) state
    const obs = this.getObservation();
    if (obs.some(v => isNaN(v))) {
      console.warn('NaN in observation, skipping');
      return;
    }

    // 3. Run ONNX inference (synchronous via await)
    try {
      const inputTensor = new ort.Tensor('float32', obs, [1, obs.length]);
      const feeds = {};
      feeds[this.inputName] = inputTensor;

      const results = await this.session.run(feeds);
      const output = results[this.outputName];
      if (!output) return;

      const action = new Float32Array(output.data);

      // 4. Update action history AFTER obs and inference (matches Python)
      this.lastLastLastAction.set(this.lastLastAction);
      this.lastLastAction.set(this.lastAction);
      this.lastAction.set(action);

      // 5. Compute new motor targets
      for (let i = 0; i < this.numDofs; i++) {
        this.motorTargets[i] = this.defaultActuator[i] + action[i] * this.actionScale;
      }

      // 6. Velocity clamp (matches Python np.clip)
      for (let i = 0; i < this.numDofs; i++) {
        const maxChange = this.maxMotorVelocity * this.ctrlDt;
        const diff = this.motorTargets[i] - this.prevMotorTargets[i];
        if (Math.abs(diff) > maxChange) {
          this.motorTargets[i] = this.prevMotorTargets[i] + Math.sign(diff) * maxChange;
        }
        this.prevMotorTargets[i] = this.motorTargets[i];
      }

      // 7. Apply to ctrl (persists for next decimation steps)
      const ctrl = this.data.ctrl;
      for (let i = 0; i < Math.min(this.numDofs, ctrl.length); i++) {
        ctrl[i] = this.motorTargets[i];
      }
    } catch (e) {
      console.error('ONNX inference error:', e);
    }
  }

  reset() {
    if (this.lastAction) this.lastAction.fill(0);
    if (this.lastLastAction) this.lastLastAction.fill(0);
    if (this.lastLastLastAction) this.lastLastLastAction.fill(0);

    this.imitationI = 0;
    this.imitationPhase = [0, 0]; // Updated before first use
    this.commands = [0, 0, 0, 0, 0, 0, 0];
    this.stepCounter = 0;

    if (this.motorTargets && this.defaultActuator) {
      this.motorTargets.set(this.defaultActuator);
      this.prevMotorTargets.set(this.defaultActuator);
    }

    // Reset ctrl to default
    if (this.data && this.data.ctrl) {
      const ctrl = this.data.ctrl;
      for (let i = 0; i < Math.min(this.numDofs, ctrl.length); i++) {
        ctrl[i] = this.defaultActuator[i];
      }
    }
  }
}
