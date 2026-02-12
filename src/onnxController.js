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
    this.defaultForwardCommand = 0.0;
    this.defaultNeckPitchCommand = 0.55;
    this.startupNeckPitchCommand = 0.8;
    this.startupAssistDuration = 2.2;

    // Imitation phase: period=0.54s, fps=50Hz → nb_steps_in_period=27
    // Verified from upstream polynomial_coefficients.pkl data
    this.imitationI = 0;
    this.nbStepsInPeriod = 27;
    this.imitationPhase = [0, 0];

    // Step counter - incremented AFTER each mj_step (matches Python)
    this.stepCounter = 0;

    // Policy step counter for diagnostics
    this.policyStepCount = 0;
    this._policyRunning = false;

    // Sensor addresses
    this.gyroAddr = -1;
    this.accelAddr = -1;

    // Body IDs for contact detection (matching Python check_contact)
    this.leftFootBodyId = -1;
    this.rightFootBodyId = -1;
    this.floorBodyId = -1;

    // Joint index mapping (handles backlash joints that shift qpos layout)
    this.qposIndices = null;
    this.qvelIndices = null;
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
      console.log(`ONNX input: "${this.inputName}", output: "${this.outputName}"`);

      this.numDofs = this.model.nu;
      this.initState();
      this.findSensorAddresses();
      this.findBodyIds();
      this.findJointIndices();
      this.logActuatorDiagnostics();

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

    // Start with a small forward velocity so the duck walks automatically
    this.commands[0] = this.defaultForwardCommand;
    // Use a higher neck command only at startup, then blend to cruise value.
    // commands[3]=neck_pitch, commands[4]=head_pitch
    this.commands[3] = this.startupNeckPitchCommand;

    console.log('Default actuator:', Array.from(this.defaultActuator).map(v => v.toFixed(3)));
    console.log('Default command: forward velocity =', this.commands[0]);
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
      }
    }

    // Fallback by sensor type (correct MuJoCo enum: GYRO=3, ACCELEROMETER=1)
    if (this.gyroAddr < 0 || this.accelAddr < 0) {
      for (let i = 0; i < nsensor; i++) {
        const type = this.model.sensor_type[i];
        const adr = this.model.sensor_adr[i];
        if (type === 3 && this.gyroAddr < 0) this.gyroAddr = adr;  // mjSENS_GYRO=3
        if (type === 1 && this.accelAddr < 0) this.accelAddr = adr; // mjSENS_ACCELEROMETER=1
      }
    }

    if (this.gyroAddr < 0) this.gyroAddr = 0;
    if (this.accelAddr < 0) this.accelAddr = 6;

    console.log(`Sensors: gyro@${this.gyroAddr}, accel@${this.accelAddr}`);
  }

  findBodyIds() {
    // Find body IDs for contact detection (matching Python: check_contact)
    try {
      // mjOBJ_BODY = 1
      this.leftFootBodyId = this.mujoco.mj_name2id(this.model, 1, 'foot_assembly');
      this.rightFootBodyId = this.mujoco.mj_name2id(this.model, 1, 'foot_assembly_2');
      this.floorBodyId = this.mujoco.mj_name2id(this.model, 1, 'floor');
      console.log(`Bodies: leftFoot=${this.leftFootBodyId}, rightFoot=${this.rightFootBodyId}, floor=${this.floorBodyId}`);
    } catch (e) {
      console.warn('Could not find body IDs for contact detection:', e);
    }
  }

  findJointIndices() {
    // Map actuator index → qpos/qvel address via joint lookup.
    // Critical for backlash model where extra joints shift the qpos layout.
    // Actuator names match joint names in the XML.
    const jointNames = [
      'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
      'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
      'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle'
    ];

    this.qposIndices = new Int32Array(this.numDofs);
    this.qvelIndices = new Int32Array(this.numDofs);

    let success = false;
    try {
      if (this.model.jnt_qposadr && this.model.jnt_dofadr) {
        for (let i = 0; i < this.numDofs; i++) {
          // mjOBJ_JOINT = 3
          const jointId = this.mujoco.mj_name2id(this.model, 3, jointNames[i]);
          if (jointId >= 0) {
            this.qposIndices[i] = this.model.jnt_qposadr[jointId];
            this.qvelIndices[i] = this.model.jnt_dofadr[jointId];
          } else {
            throw new Error(`Joint "${jointNames[i]}" not found`);
          }
        }
        success = true;
      }
    } catch (e) {
      console.warn('Joint index lookup failed, using fallback:', e);
    }

    // Fallback: simple offset (works for non-backlash model)
    if (!success) {
      for (let i = 0; i < this.numDofs; i++) {
        this.qposIndices[i] = 7 + i;
        this.qvelIndices[i] = 6 + i;
      }
    }

    console.log('Joint qpos indices:', Array.from(this.qposIndices));
    console.log('Joint qvel indices:', Array.from(this.qvelIndices));
  }

  logActuatorDiagnostics() {
    const n = this.numDofs;
    try {
      console.log('=== ACTUATOR CONFIG ===');
      console.log('timestep:', this.model.opt.timestep, '| nu:', n);

      if (this.model.actuator_biastype) {
        const bt = Array.from(this.model.actuator_biastype).slice(0, n);
        console.log('biastype:', bt, '(expect all 1 = affine/position)');
        if (bt.some(v => v !== 1)) {
          console.error('CRITICAL: biastype is NOT all 1! Actuators may not work as position servos.');
        }
      }
      if (this.model.actuator_gainprm) {
        console.log('kp:', this.model.actuator_gainprm[0].toFixed(2));
      }
      if (this.model.actuator_biasprm) {
        console.log('biasprm[0:3]:', [
          this.model.actuator_biasprm[0],
          this.model.actuator_biasprm[1],
          this.model.actuator_biasprm[2]
        ].map(v => v.toFixed(2)));
      }
      console.log('=== END CONFIG ===');
    } catch (e) {
      console.warn('Actuator diagnostics failed:', e);
    }
  }

  /**
   * Fire-and-forget async policy execution.
   * Updates motorTargets when inference completes.
   */
  runPolicyAsync() {
    if (this._policyRunning) return;
    this._policyRunning = true;
    this.runPolicy().then(() => {
      this._policyRunning = false;
    }).catch((e) => {
      console.error('Policy error:', e);
      this._policyRunning = false;
    });
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
    // Match Python inference: accelerometer[0] += 1.3
    return [
      this.data.sensordata[this.accelAddr] + 1.3,
      this.data.sensordata[this.accelAddr + 1],
      this.data.sensordata[this.accelAddr + 2]
    ];
  }

  getActuatorJointsQpos() {
    // Uses qposIndices for correct mapping (handles backlash joints)
    const angles = new Float32Array(this.numDofs);
    for (let i = 0; i < this.numDofs; i++) {
      angles[i] = this.data.qpos[this.qposIndices[i]];
    }
    return angles;
  }

  getActuatorJointsQvel() {
    // Uses qvelIndices for correct mapping (handles backlash joints)
    const vels = new Float32Array(this.numDofs);
    for (let i = 0; i < this.numDofs; i++) {
      vels[i] = this.data.qvel[this.qvelIndices[i]];
    }
    return vels;
  }

  getFeetContacts() {
    // Match Python exactly: check_contact(data, "foot_assembly", "floor")
    // Iterates data.contact array and checks geom body IDs
    if (this.leftFootBodyId >= 0 && this.rightFootBodyId >= 0 && this.floorBodyId >= 0) {
      let leftContact = 0.0;
      let rightContact = 0.0;

      const ncon = this.data.ncon;
      for (let i = 0; i < ncon; i++) {
        try {
          const contact = this.data.contact.get(i);
          if (!contact) continue;
          const body1 = this.model.geom_bodyid[contact.geom1];
          const body2 = this.model.geom_bodyid[contact.geom2];

          // Check left foot - floor contact
          if ((body1 === this.leftFootBodyId && body2 === this.floorBodyId) ||
              (body1 === this.floorBodyId && body2 === this.leftFootBodyId)) {
            leftContact = 1.0;
          }
          // Check right foot - floor contact
          if ((body1 === this.rightFootBodyId && body2 === this.floorBodyId) ||
              (body1 === this.floorBodyId && body2 === this.rightFootBodyId)) {
            rightContact = 1.0;
          }
        } catch (e) {
          break;
        }
      }

      return [leftContact, rightContact];
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

    this.policyStepCount++;

    // Startup assist: keep neck up briefly, then smoothly blend to cruise command.
    const simTime = this.stepCounter * this.simDt;
    const q = this.data.qpos;
    const pitchDeg = Math.asin(2 * (q[3] * q[5] - q[6] * q[4])) * 180 / Math.PI;
    const backLeanAssist = Math.max(0, Math.min(0.25, (-pitchDeg - 3) * 0.01));
    let neckTarget = this.defaultNeckPitchCommand + backLeanAssist;
    if (simTime < this.startupAssistDuration) {
      const a = simTime / this.startupAssistDuration;
      const blended = this.startupNeckPitchCommand * (1 - a) + this.defaultNeckPitchCommand * a;
      neckTarget = Math.max(neckTarget, blended + backLeanAssist);
    }
    this.commands[3] = Math.max(this.defaultNeckPitchCommand, Math.min(0.9, Math.max(this.commands[3], neckTarget)));

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

    // Diagnostic for first policy step
    if (this.policyStepCount === 1) {
      const quat = Array.from(this.data.qpos).slice(3,7);
      const contacts = this.getFeetContacts();
      const gyro = this.getGyro();
      const accel = this.getAccelerometer();
      const rawAccel = [
        this.data.sensordata[this.accelAddr],
        this.data.sensordata[this.accelAddr + 1],
        this.data.sensordata[this.accelAddr + 2]
      ];
      console.log('=== FIRST POLICY STEP ===');
      console.log(`Pos: [${Array.from(this.data.qpos).slice(0,3).map(v=>v.toFixed(4))}]`);
      console.log(`Quat: [${quat.map(v=>v.toFixed(4))}]`);
      console.log(`Gyro: [${gyro.map(v=>v.toFixed(4))}]`);
      console.log(`Accel (raw): [${rawAccel.map(v=>v.toFixed(4))}]`);
      console.log(`Accel (+1.3 bias): [${accel.map(v=>v.toFixed(4))}]`);
      console.log(`Contacts: [${contacts}] ncon=${this.data.ncon}`);
      console.log(`Commands: [${this.commands.map(v=>v.toFixed(3))}]`);
      console.log(`Obs length: ${obs.length}`);
      console.log(`Solver iterations: ${this.model.opt.iterations}`);
      console.log('========================');
    }

    // Brief logging for first 30 policy steps
    if (this.policyStepCount <= 30) {
      const h = (this.data.qpos[2] || 0).toFixed(4);
      const quat = [this.data.qpos[3], this.data.qpos[4], this.data.qpos[5], this.data.qpos[6]];
      const pitch = Math.asin(2*(quat[0]*quat[2] - quat[3]*quat[1])) * 180/Math.PI;
      const contacts = this.getFeetContacts();
      console.log(`[Policy #${this.policyStepCount}] H=${h} pitch=${pitch.toFixed(1)}° contacts=[${contacts}] cmd=[${this.commands[0].toFixed(2)},${this.commands[1].toFixed(2)}]`);
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

      // Check if actions are in expected range (tanh should give [-1,1])
      if (this.policyStepCount <= 3) {
        const maxAct = Math.max(...action);
        const minAct = Math.min(...action);
        console.log(`  action range=[${minAct.toFixed(3)}, ${maxAct.toFixed(3)}]`);
      }

      // 4. Update action history AFTER obs and inference (matches Python)
      this.lastLastLastAction.set(this.lastLastAction);
      this.lastLastAction.set(this.lastAction);
      this.lastAction.set(action);

      // 5. Compute new motor targets
      for (let i = 0; i < this.numDofs; i++) {
        this.motorTargets[i] = this.defaultActuator[i] + action[i] * this.actionScale;
      }

      // Hard safety floor for neck/head targets to avoid immediate backward collapse.
      // actuator order: ... neck_pitch(5), head_pitch(6), ...
      this.motorTargets[5] = Math.max(this.motorTargets[5], this.defaultNeckPitchCommand);
      this.motorTargets[6] = Math.max(this.motorTargets[6], 0.1);

      // 6. Velocity clamp (matches Python np.clip)
      for (let i = 0; i < this.numDofs; i++) {
        const maxChange = this.maxMotorVelocity * this.ctrlDt;
        const diff = this.motorTargets[i] - this.prevMotorTargets[i];
        if (Math.abs(diff) > maxChange) {
          this.motorTargets[i] = this.prevMotorTargets[i] + Math.sign(diff) * maxChange;
        }
        this.prevMotorTargets[i] = this.motorTargets[i];
      }

      // Apply to ctrl (persists for next decimation steps)
      const ctrl = this.data.ctrl;
      for (let i = 0; i < Math.min(this.numDofs, ctrl.length); i++) {
        ctrl[i] = this.motorTargets[i];
      }

      if (this.policyStepCount <= 3) {
        const tgtStr = Array.from(this.motorTargets).slice(0, 5).map(v => v.toFixed(3));
        console.log(`  targets=[${tgtStr}...]`);
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
    this.imitationPhase = [0, 0];
    this.commands = [this.defaultForwardCommand, 0, 0, this.startupNeckPitchCommand, 0, 0, 0];
    this.stepCounter = 0;
    this.policyStepCount = 0;

    if (this.motorTargets && this.defaultActuator) {
      this.motorTargets.set(this.defaultActuator);
      this.prevMotorTargets.set(this.defaultActuator);

      // Restore ctrl to default (critical: old ctrl values persist otherwise)
      const ctrl = this.data.ctrl;
      for (let i = 0; i < Math.min(this.numDofs, ctrl.length); i++) {
        ctrl[i] = this.defaultActuator[i];
      }
    }
    this._policyRunning = false;
  }
}
