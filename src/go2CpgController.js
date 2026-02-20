/**
 * Go2 CPG (Central Pattern Generator) Controller
 *
 * Implements improved walking gait with swing/stance separation and
 * a trick state machine for acrobatic maneuvers (jump, frontflip, backflip, sideroll).
 *
 * Go2 joint layout (12 actuators, 3 per leg):
 *   0: FL_hip (abduction)   1: FL_thigh   2: FL_calf (knee)
 *   3: FR_hip               4: FR_thigh   5: FR_calf
 *   6: RL_hip               7: RL_thigh   8: RL_calf
 *   9: RR_hip              10: RR_thigh  11: RR_calf
 *
 * Home pose: hip=0, thigh=0.9, calf=-1.8, standing height ~0.27m
 */

// ── Leg indices ──────────────────────────────────────────────────────
const FL = 0, FR = 1, RL = 2, RR = 3;
const HIP = 0, THIGH = 1, CALF = 2;

// Actuator index for a given leg and joint
function actuatorIdx(leg, joint) { return leg * 3 + joint; }

// ── Home position ────────────────────────────────────────────────────
const HOME_HIP   =  0.0;
const HOME_THIGH =  0.9;
const HOME_CALF  = -1.8;

// ── PD gains (base values, scaled by trick phase) ────────────────────
const KP_HIP   = 40;
const KP_THIGH = 40;
const KP_CALF  = 50;
const KD_HIP   = 2.0;
const KD_THIGH = 2.0;
const KD_CALF  = 2.5;

// ── Trick phases ─────────────────────────────────────────────────────
const PHASE = {
  IDLE:    'idle',
  CROUCH:  'crouch',
  LAUNCH:  'launch',
  AIR:     'air',
  LAND:    'land',
  RECOVER: 'recover',
};

// Gain scaling per trick phase
const GAIN_SCALE = {
  [PHASE.IDLE]:    { kp: 1.0, kd: 1.0 },
  [PHASE.CROUCH]:  { kp: 1.5, kd: 1.0 },
  [PHASE.LAUNCH]:  { kp: 2.0, kd: 0.5 },
  [PHASE.AIR]:     { kp: 0.3, kd: 0.5 },
  [PHASE.LAND]:    { kp: 0.8, kd: 2.5 },
  [PHASE.RECOVER]: { kp: 1.0, kd: 1.0 },
};

// ── Trick definitions ────────────────────────────────────────────────
// Each trick is a sequence of phases with step durations and per-leg targets
// Targets: [hip, thigh, calf] for each leg [FL, FR, RL, RR]

const TRICKS = {
  jump: {
    phases: [
      {
        phase: PHASE.CROUCH, steps: 60,
        targets: {
          FL: [0, 1.4, -2.6], FR: [0, 1.4, -2.6],
          RL: [0, 1.4, -2.6], RR: [0, 1.4, -2.6],
        }
      },
      {
        phase: PHASE.LAUNCH, steps: 40,
        targets: {
          FL: [0, 0.3, -0.85], FR: [0, 0.3, -0.85],
          RL: [0, 0.3, -0.85], RR: [0, 0.3, -0.85],
        }
      },
      {
        phase: PHASE.AIR, steps: 150,
        targets: {
          FL: [0, 1.2, -2.5], FR: [0, 1.2, -2.5],
          RL: [0, 1.2, -2.5], RR: [0, 1.2, -2.5],
        }
      },
      {
        phase: PHASE.LAND, steps: 50,
        targets: {
          FL: [0, 1.3, -2.4], FR: [0, 1.3, -2.4],
          RL: [0, 1.3, -2.4], RR: [0, 1.3, -2.4],
        }
      },
      {
        phase: PHASE.RECOVER, steps: 100,
        targets: {
          FL: [HOME_HIP, HOME_THIGH, HOME_CALF],
          FR: [HOME_HIP, HOME_THIGH, HOME_CALF],
          RL: [HOME_HIP, HOME_THIGH, HOME_CALF],
          RR: [HOME_HIP, HOME_THIGH, HOME_CALF],
        }
      },
    ]
  },

  frontflip: {
    phases: [
      {
        phase: PHASE.CROUCH, steps: 50,
        targets: {
          FL: [0, 1.5, -2.65], FR: [0, 1.5, -2.65],
          RL: [0, 1.3, -2.5],  RR: [0, 1.3, -2.5],
        }
      },
      {
        phase: PHASE.LAUNCH, steps: 40,
        targets: {
          FL: [0, 0.6, -1.0],   FR: [0, 0.6, -1.0],
          RL: [0, -0.2, -0.85], RR: [0, -0.2, -0.85],
        }
      },
      {
        phase: PHASE.AIR, steps: 225,
        targets: {
          FL: [0, 1.5, -2.7], FR: [0, 1.5, -2.7],
          RL: [0, 1.5, -2.7], RR: [0, 1.5, -2.7],
        }
      },
      {
        phase: PHASE.LAND, steps: 50,
        targets: {
          FL: [0, 1.3, -2.4], FR: [0, 1.3, -2.4],
          RL: [0, 1.3, -2.4], RR: [0, 1.3, -2.4],
        }
      },
      {
        phase: PHASE.RECOVER, steps: 100,
        targets: {
          FL: [HOME_HIP, HOME_THIGH, HOME_CALF],
          FR: [HOME_HIP, HOME_THIGH, HOME_CALF],
          RL: [HOME_HIP, HOME_THIGH, HOME_CALF],
          RR: [HOME_HIP, HOME_THIGH, HOME_CALF],
        }
      },
    ]
  },

  backflip: {
    phases: [
      {
        phase: PHASE.CROUCH, steps: 50,
        targets: {
          FL: [0, 1.3, -2.5],  FR: [0, 1.3, -2.5],
          RL: [0, 1.5, -2.65], RR: [0, 1.5, -2.65],
        }
      },
      {
        phase: PHASE.LAUNCH, steps: 40,
        targets: {
          FL: [0, -0.5, -0.85], FR: [0, -0.5, -0.85],
          RL: [0, 0.6, -1.0],   RR: [0, 0.6, -1.0],
        }
      },
      {
        phase: PHASE.AIR, steps: 225,
        targets: {
          FL: [0, 1.5, -2.7], FR: [0, 1.5, -2.7],
          RL: [0, 1.5, -2.7], RR: [0, 1.5, -2.7],
        }
      },
      {
        phase: PHASE.LAND, steps: 50,
        targets: {
          FL: [0, 1.3, -2.4], FR: [0, 1.3, -2.4],
          RL: [0, 1.3, -2.4], RR: [0, 1.3, -2.4],
        }
      },
      {
        phase: PHASE.RECOVER, steps: 100,
        targets: {
          FL: [HOME_HIP, HOME_THIGH, HOME_CALF],
          FR: [HOME_HIP, HOME_THIGH, HOME_CALF],
          RL: [HOME_HIP, HOME_THIGH, HOME_CALF],
          RR: [HOME_HIP, HOME_THIGH, HOME_CALF],
        }
      },
    ]
  },

  sideroll: {
    phases: [
      {
        phase: PHASE.CROUCH, steps: 50,
        targets: {
          FL: [0, 1.4, -2.6], FR: [0, 1.4, -2.6],
          RL: [0, 1.4, -2.6], RR: [0, 1.4, -2.6],
        }
      },
      {
        phase: PHASE.LAUNCH, steps: 40,
        targets: {
          FL: [0.8, 0.3, -0.85],  FR: [-0.8, 0.6, -1.2],
          RL: [0.8, 0.3, -0.85],  RR: [-0.8, 0.6, -1.2],
        }
      },
      {
        phase: PHASE.AIR, steps: 200,
        targets: {
          FL: [0, 1.2, -2.5], FR: [0, 1.2, -2.5],
          RL: [0, 1.2, -2.5], RR: [0, 1.2, -2.5],
        }
      },
      {
        phase: PHASE.LAND, steps: 50,
        targets: {
          FL: [0, 1.3, -2.4], FR: [0, 1.3, -2.4],
          RL: [0, 1.3, -2.4], RR: [0, 1.3, -2.4],
        }
      },
      {
        phase: PHASE.RECOVER, steps: 100,
        targets: {
          FL: [HOME_HIP, HOME_THIGH, HOME_CALF],
          FR: [HOME_HIP, HOME_THIGH, HOME_CALF],
          RL: [HOME_HIP, HOME_THIGH, HOME_CALF],
          RR: [HOME_HIP, HOME_THIGH, HOME_CALF],
        }
      },
    ]
  },
};

// Leg name keys for target lookup
const LEG_KEYS = ['FL', 'FR', 'RL', 'RR'];

// ── Gait parameters ──────────────────────────────────────────────────
const DUTY_FACTOR  = 0.6;   // 60% stance, 40% swing
const SWING_HEIGHT = 0.35;  // rad amplitude for foot lift during swing
const BASE_FREQ    = 1.8;   // Hz at idle
const MAX_FREQ     = 3.3;   // Hz at full speed
const THIGH_AMPLITUDE = 0.25; // rad forward/backward swing
const CALF_STANCE_OFFSET = -0.1; // slight extra bend during stance push

// Trot gait phase offsets: diagonal legs in phase, opposite pair offset by pi
const GAIT_OFFSETS = [0, Math.PI, Math.PI, 0]; // FL, FR, RL, RR

export class Go2CpgController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;

    // Sim params
    this.simDt = 0.002;
    this.decimation = 10;   // control at 50 Hz (every 10 steps)
    this.stepCounter = 0;

    // CPG state
    this.phase = 0;        // gait phase [0, 2*PI)
    this.cpgTime = 0;

    // Movement commands
    this.commands = { linX: 0, linY: 0, angZ: 0 };

    // Joint targets (12 values)
    this.targets = new Float64Array(12);

    // Trick state machine
    this.trickPhase = PHASE.IDLE;
    this.trickName = null;
    this.trickPhaseIdx = 0;
    this.trickStepCount = 0;
    this.trickStartTargets = new Float64Array(12); // snapshot at trick start

    // Foot contact
    this.footGeomIds = [-1, -1, -1, -1]; // FL, FR, RL, RR
    this.floorBodyId = -1;

    // Joint address mapping
    this.qposAddrs = new Int32Array(12);
    this.qvelAddrs = new Int32Array(12);

    // Initialize
    this.findJointAddresses();
    this.findFootGeoms();
    this.resetTargetsToHome();
  }

  findJointAddresses() {
    const jointNames = [
      'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
      'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
      'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
      'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    ];

    for (let i = 0; i < 12; i++) {
      try {
        // mjOBJ_JOINT = 3
        const jntId = this.mujoco.mj_name2id(this.model, 3, jointNames[i]);
        if (jntId >= 0) {
          this.qposAddrs[i] = this.model.jnt_qposadr[jntId];
          this.qvelAddrs[i] = this.model.jnt_dofadr[jntId];
        } else {
          // Fallback: freejoint has 7 qpos, then 12 joint qpos
          this.qposAddrs[i] = 7 + i;
          this.qvelAddrs[i] = 6 + i;
        }
      } catch (e) {
        this.qposAddrs[i] = 7 + i;
        this.qvelAddrs[i] = 6 + i;
      }
    }
  }

  findFootGeoms() {
    const footNames = ['FL', 'FR', 'RL', 'RR'];
    for (let i = 0; i < 4; i++) {
      try {
        // mjOBJ_GEOM = 5
        const gid = this.mujoco.mj_name2id(this.model, 5, footNames[i]);
        if (gid >= 0) this.footGeomIds[i] = gid;
      } catch (e) { /* ignore */ }
    }
    try {
      // mjOBJ_BODY = 1
      this.floorBodyId = this.mujoco.mj_name2id(this.model, 1, 'floor');
    } catch (e) {
      // Try geom-based floor detection
      try {
        const floorGeomId = this.mujoco.mj_name2id(this.model, 5, 'floor');
        if (floorGeomId >= 0) {
          this.floorBodyId = this.model.geom_bodyid[floorGeomId];
        }
      } catch (e2) { /* ignore */ }
    }
  }

  getFootContacts() {
    const contacts = [0, 0, 0, 0]; // FL, FR, RL, RR
    const ncon = this.data.ncon || 0;

    for (let c = 0; c < ncon; c++) {
      try {
        const contact = this.data.contact.get(c);
        const g1 = contact.geom1;
        const g2 = contact.geom2;
        const b1 = this.model.geom_bodyid[g1];
        const b2 = this.model.geom_bodyid[g2];

        for (let f = 0; f < 4; f++) {
          if (this.footGeomIds[f] < 0) continue;
          const footBodyId = this.model.geom_bodyid[this.footGeomIds[f]];
          const isFootG1 = (b1 === footBodyId);
          const isFootG2 = (b2 === footBodyId);
          if (!isFootG1 && !isFootG2) continue;

          // Check if other geom is floor
          const otherBody = isFootG1 ? b2 : b1;
          if (this.floorBodyId >= 0 && otherBody === this.floorBodyId) {
            contacts[f] = 1;
          } else if (this.floorBodyId < 0) {
            // If we can't identify floor, accept any contact
            contacts[f] = 1;
          }
        }
      } catch (e) { break; }
    }

    // Fallback: height-based heuristic
    if (contacts[0] + contacts[1] + contacts[2] + contacts[3] === 0) {
      const h = this.data.qpos[2] || 0;
      if (h < 0.20) return [1, 1, 1, 1];
    }

    return contacts;
  }

  resetTargetsToHome() {
    for (let leg = 0; leg < 4; leg++) {
      this.targets[actuatorIdx(leg, HIP)]   = HOME_HIP;
      this.targets[actuatorIdx(leg, THIGH)] = HOME_THIGH;
      this.targets[actuatorIdx(leg, CALF)]  = HOME_CALF;
    }
  }

  setCommand(linX, linY, angZ) {
    this.commands.linX = linX;
    this.commands.linY = linY;
    this.commands.angZ = angZ;
  }

  triggerTrick(name) {
    if (this.trickPhase !== PHASE.IDLE) return; // already doing a trick
    if (!TRICKS[name]) return;

    this.trickName = name;
    this.trickPhaseIdx = 0;
    this.trickStepCount = 0;
    this.trickPhase = TRICKS[name].phases[0].phase;

    // Snapshot current targets for smooth interpolation
    this.trickStartTargets.set(this.targets);

    console.log(`Go2 trick: ${name}`);
  }

  // ── PD torque computation with gain scaling ────────────────────────
  pdTorque(actuatorId, targetPos) {
    const qpos = this.data.qpos[this.qposAddrs[actuatorId]];
    const qvel = this.data.qvel[this.qvelAddrs[actuatorId]];

    const joint = actuatorId % 3;
    let kp = joint === CALF ? KP_CALF : (joint === THIGH ? KP_THIGH : KP_HIP);
    let kd = joint === CALF ? KD_CALF : (joint === THIGH ? KD_THIGH : KD_HIP);

    // Scale gains by trick phase
    const scale = GAIN_SCALE[this.trickPhase];
    kp *= scale.kp;
    kd *= scale.kd;

    return kp * (targetPos - qpos) - kd * qvel;
  }

  // ── Walking CPG ────────────────────────────────────────────────────
  updateWalkingCPG() {
    const { linX, linY, angZ } = this.commands;
    const speedMag = Math.sqrt(linX * linX + linY * linY);

    // Speed-responsive frequency
    const freq = BASE_FREQ + speedMag * (MAX_FREQ - BASE_FREQ) / 0.15;
    const dt = this.simDt * this.decimation; // control period
    this.cpgTime += dt;
    this.phase = (this.cpgTime * freq * 2 * Math.PI) % (2 * Math.PI);

    for (let leg = 0; leg < 4; leg++) {
      const legPhase = (this.phase + GAIT_OFFSETS[leg]) % (2 * Math.PI);
      const normalized = legPhase / (2 * Math.PI); // [0, 1)

      // Determine stance vs swing
      const isStance = normalized < DUTY_FACTOR;

      let hipTarget = HOME_HIP;
      let thighTarget = HOME_THIGH;
      let calfTarget = HOME_CALF;

      if (speedMag > 0.005 || Math.abs(angZ) > 0.05) {
        // Direction-dependent amplitude
        let fwdAmp = linX * THIGH_AMPLITUDE / 0.10;
        let latAmp = linY * 0.1 / 0.15;

        // Differential turning: adjust left/right amplitudes
        const isLeft = (leg === FL || leg === RL);
        let turnScale = 1.0;
        if (angZ !== 0) {
          turnScale = isLeft
            ? 1.0 + angZ * 0.5  // positive angZ = turn left = left legs slower
            : 1.0 - angZ * 0.5;
          turnScale = Math.max(0.2, Math.min(2.0, turnScale));
        }

        if (isStance) {
          // Stance: foot pushes backward, thigh moves back
          const stanceProgress = normalized / DUTY_FACTOR; // [0, 1]
          thighTarget = HOME_THIGH + fwdAmp * turnScale * (0.5 - stanceProgress);
          calfTarget = HOME_CALF + CALF_STANCE_OFFSET;
          hipTarget = HOME_HIP + latAmp * (0.5 - stanceProgress);
        } else {
          // Swing: foot lifts, thigh swings forward
          const swingProgress = (normalized - DUTY_FACTOR) / (1 - DUTY_FACTOR); // [0, 1]
          // Bell-curve lift: sin gives smooth up-down
          const lift = Math.sin(swingProgress * Math.PI) * SWING_HEIGHT;
          thighTarget = HOME_THIGH + fwdAmp * turnScale * (-0.5 + swingProgress) - lift * 0.5;
          calfTarget = HOME_CALF - lift; // bend knee more during swing for clearance
          hipTarget = HOME_HIP + latAmp * (-0.5 + swingProgress);
        }
      }

      this.targets[actuatorIdx(leg, HIP)]   = hipTarget;
      this.targets[actuatorIdx(leg, THIGH)] = thighTarget;
      this.targets[actuatorIdx(leg, CALF)]  = calfTarget;
    }
  }

  // ── Trick state machine ────────────────────────────────────────────
  updateTrickStateMachine() {
    const trick = TRICKS[this.trickName];
    if (!trick) { this.trickPhase = PHASE.IDLE; return; }

    const phaseSpec = trick.phases[this.trickPhaseIdx];
    if (!phaseSpec) { this.trickPhase = PHASE.IDLE; this.trickName = null; return; }

    this.trickStepCount++;
    this.trickPhase = phaseSpec.phase;

    // Smooth interpolation toward phase targets
    const progress = Math.min(1.0, this.trickStepCount / phaseSpec.steps);
    // Ease-in-out
    const t = progress < 0.5
      ? 2 * progress * progress
      : 1 - Math.pow(-2 * progress + 2, 2) / 2;

    for (let leg = 0; leg < 4; leg++) {
      const legKey = LEG_KEYS[leg];
      const target = phaseSpec.targets[legKey];
      for (let j = 0; j < 3; j++) {
        const idx = actuatorIdx(leg, j);
        // Interpolate from start snapshot to phase target
        this.targets[idx] = this.trickStartTargets[idx] + (target[j] - this.trickStartTargets[idx]) * t;
      }
    }

    // Early landing detection during AIR phase
    if (phaseSpec.phase === PHASE.AIR && this.trickStepCount > 30) {
      const contacts = this.getFootContacts();
      const numContacts = contacts[0] + contacts[1] + contacts[2] + contacts[3];
      if (numContacts >= 2) {
        // Land early
        this.advanceTrickPhase();
        return;
      }
    }

    // Advance to next phase when step count exceeds duration
    if (this.trickStepCount >= phaseSpec.steps) {
      this.advanceTrickPhase();
    }
  }

  advanceTrickPhase() {
    this.trickPhaseIdx++;
    this.trickStepCount = 0;
    // Snapshot current targets as new start
    this.trickStartTargets.set(this.targets);

    const trick = TRICKS[this.trickName];
    if (!trick || this.trickPhaseIdx >= trick.phases.length) {
      // Trick complete
      this.trickPhase = PHASE.IDLE;
      this.trickName = null;
      this.trickPhaseIdx = 0;
      this.resetTargetsToHome();
    } else {
      this.trickPhase = trick.phases[this.trickPhaseIdx].phase;
    }
  }

  // ── Main control step (called every decimation steps) ──────────────
  step() {
    if (!this.enabled) return;

    if (this.trickPhase === PHASE.IDLE) {
      this.updateWalkingCPG();
    } else {
      this.updateTrickStateMachine();
    }

    // Apply PD torques
    for (let i = 0; i < 12; i++) {
      this.data.ctrl[i] = this.pdTorque(i, this.targets[i]);
    }
  }

  reset() {
    this.phase = 0;
    this.cpgTime = 0;
    this.stepCounter = 0;
    this.trickPhase = PHASE.IDLE;
    this.trickName = null;
    this.trickPhaseIdx = 0;
    this.trickStepCount = 0;
    this.resetTargetsToHome();

    // Apply home ctrl directly
    for (let i = 0; i < 12; i++) {
      this.data.ctrl[i] = this.pdTorque(i, this.targets[i]);
    }
  }

  // Get current state info for UI
  getState() {
    return {
      trickPhase: this.trickPhase,
      trickName: this.trickName,
      height: (this.data.qpos[2] || 0).toFixed(3),
      contacts: this.getFootContacts(),
    };
  }
}
