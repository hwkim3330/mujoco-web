/**
 * ONNX-based Robot Controller for OpenDuck Mini
 * Loads trained policy and controls the robot based on observations
 */

export class OnnxController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.session = null;
    this.enabled = false;

    // Action parameters
    this.actionScale = 0.25;
    this.maxMotorVelocity = 5.24;
    this.dofVelScale = 0.05;

    // State
    this.lastAction = new Float32Array(14);
    this.lastLastAction = new Float32Array(14);
    this.lastLastLastAction = new Float32Array(14);
    this.motorTargets = null;
    this.prevMotorTargets = null;
    this.imitationPhase = [0, 0];
    this.imitationI = 0;
    this.nbStepsInPeriod = 50; // Approximate from walking gait

    // Commands [lin_x, lin_y, ang_z, height, step_freq, phase_offset, swing_height]
    this.commands = [0, 0, 0, 0, 0, 0, 0];

    // Default actuator positions (from config)
    this.defaultActuator = null;
  }

  async loadModel(url) {
    try {
      // Use ONNX Runtime Web
      if (typeof ort === 'undefined') {
        console.warn('ONNX Runtime not loaded. Add ort.min.js to use ONNX inference.');
        return false;
      }

      this.session = await ort.InferenceSession.create(url);
      console.log('ONNX model loaded:', url);

      // Initialize default actuator positions
      this.initDefaultActuator();

      return true;
    } catch (e) {
      console.error('Failed to load ONNX model:', e);
      return false;
    }
  }

  initDefaultActuator() {
    // Get default positions from model keyframe or zeros
    const numActuators = this.model.nu;
    this.defaultActuator = new Float32Array(numActuators);
    this.motorTargets = new Float32Array(numActuators);
    this.prevMotorTargets = new Float32Array(numActuators);

    // Try to get from keyframe "home"
    // For now, use zeros
    for (let i = 0; i < numActuators; i++) {
      this.defaultActuator[i] = 0;
      this.motorTargets[i] = 0;
      this.prevMotorTargets[i] = 0;
    }
  }

  setCommand(linX, linY, angZ) {
    this.commands[0] = Math.max(-0.15, Math.min(0.15, linX));
    this.commands[1] = Math.max(-0.2, Math.min(0.2, linY));
    this.commands[2] = Math.max(-1.0, Math.min(1.0, angZ));
  }

  getObservation() {
    // Build observation vector (101 dimensions)
    const obs = [];

    // Gyro (3) - from sensor
    const gyro = this.getSensorData('gyro', 3);
    obs.push(...gyro);

    // Accelerometer (3) - with gravity compensation
    const accel = this.getSensorData('accelerometer', 3);
    accel[0] += 1.3; // Gravity compensation
    obs.push(...accel);

    // Commands (7)
    obs.push(...this.commands);

    // Joint angles - default (14)
    const jointAngles = this.getJointAngles();
    for (let i = 0; i < 14; i++) {
      obs.push((jointAngles[i] || 0) - (this.defaultActuator[i] || 0));
    }

    // Joint velocities (14)
    const jointVel = this.getJointVelocities();
    for (let i = 0; i < 14; i++) {
      obs.push((jointVel[i] || 0) * this.dofVelScale);
    }

    // Action history (14 * 3 = 42)
    obs.push(...this.lastAction);
    obs.push(...this.lastLastAction);
    obs.push(...this.lastLastLastAction);

    // Motor targets (14)
    obs.push(...(this.motorTargets || new Float32Array(14)));

    // Foot contacts (2)
    const contacts = this.getFootContacts();
    obs.push(...contacts);

    // Phase (2)
    obs.push(...this.imitationPhase);

    return new Float32Array(obs);
  }

  getSensorData(name, size) {
    // For now, return placeholder
    // In real implementation, read from MuJoCo sensors
    return new Float32Array(size);
  }

  getJointAngles() {
    const angles = new Float32Array(14);
    const qpos = this.data.qpos;
    // Skip first 7 (base position + quaternion)
    for (let i = 0; i < Math.min(14, qpos.length - 7); i++) {
      angles[i] = qpos[7 + i] || 0;
    }
    return angles;
  }

  getJointVelocities() {
    const velocities = new Float32Array(14);
    const qvel = this.data.qvel;
    // Skip first 6 (base linear + angular velocity)
    for (let i = 0; i < Math.min(14, qvel.length - 6); i++) {
      velocities[i] = qvel[6 + i] || 0;
    }
    return velocities;
  }

  getFootContacts() {
    // Simple height-based contact detection
    const height = this.data.qpos[2] || 0;
    if (height < 0.15) {
      return [1.0, 1.0];
    }
    return [0.0, 0.0];
  }

  async step(timestep) {
    if (!this.session || !this.enabled) {
      return;
    }

    try {
      // Update phase
      this.imitationI = (this.imitationI + 1) % this.nbStepsInPeriod;
      const phase = (this.imitationI / this.nbStepsInPeriod) * 2 * Math.PI;
      this.imitationPhase[0] = Math.cos(phase);
      this.imitationPhase[1] = Math.sin(phase);

      // Get observation
      const obs = this.getObservation();

      // Run inference
      const inputTensor = new ort.Tensor('float32', obs, [1, obs.length]);
      const feeds = { obs: inputTensor };
      const results = await this.session.run(feeds);

      // Get action
      const action = results.output ? results.output.data : new Float32Array(14);

      // Update action history
      this.lastLastLastAction.set(this.lastLastAction);
      this.lastLastAction.set(this.lastAction);
      this.lastAction.set(action);

      // Apply action to motors
      this.applyAction(action, timestep);

    } catch (e) {
      console.error('ONNX inference error:', e);
    }
  }

  applyAction(action, timestep) {
    const numActuators = Math.min(action.length, this.model.nu);

    // Compute target positions
    for (let i = 0; i < numActuators; i++) {
      this.motorTargets[i] = this.defaultActuator[i] + action[i] * this.actionScale;
    }

    // Apply velocity limit
    const maxChange = this.maxMotorVelocity * timestep;
    for (let i = 0; i < numActuators; i++) {
      const diff = this.motorTargets[i] - this.prevMotorTargets[i];
      if (Math.abs(diff) > maxChange) {
        this.motorTargets[i] = this.prevMotorTargets[i] + Math.sign(diff) * maxChange;
      }
      this.prevMotorTargets[i] = this.motorTargets[i];
    }

    // Apply to MuJoCo
    const ctrl = this.data.ctrl;
    for (let i = 0; i < numActuators; i++) {
      ctrl[i] = this.motorTargets[i];
    }
  }

  reset() {
    this.lastAction.fill(0);
    this.lastLastAction.fill(0);
    this.lastLastLastAction.fill(0);
    this.imitationI = 0;
    this.commands = [0, 0, 0, 0, 0, 0, 0];

    if (this.motorTargets) {
      this.motorTargets.set(this.defaultActuator);
      this.prevMotorTargets.set(this.defaultActuator);
    }
  }
}
