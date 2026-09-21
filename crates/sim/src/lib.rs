mod config;
mod terrain;
pub use config::*;
use rapier3d::{
    control::{DynamicRayCastVehicleController, WheelTuning},
    prelude::*,
};
use serde::{Deserialize, Serialize};
pub use terrain::*;
pub const OBS_DIM: usize = 18;
pub type Observation = [f32; OBS_DIM];
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Action {
    pub throttle: f32,
    pub steering: f32,
}
impl Action {
    pub fn validate(self) -> Result<(), String> {
        if !self.throttle.is_finite()
            || !self.steering.is_finite()
            || self.throttle.abs() > 1.
            || self.steering.abs() > 1.
        {
            Err("actions must be finite and in [-1,1]".into())
        } else {
            Ok(())
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VehicleSnapshot {
    pub id: String,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub speed: f32,
    pub wheels: Vec<[f32; 3]>,
    pub materials: Vec<u32>,
    pub active: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub time: f32,
    pub vehicles: Vec<VehicleSnapshot>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepResult {
    pub observations: Vec<Observation>,
    pub rewards: Vec<f32>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub active: Vec<bool>,
    pub reasons: Vec<String>,
}
struct Vehicle {
    body: RigidBodyHandle,
    collider: ColliderHandle,
    dynamics: DynamicRayCastVehicleController,
    action: Action,
    terminated: bool,
    truncated: bool,
    reason: String,
    materials: Vec<u32>,
}
pub struct World {
    pub scenario: Scenario,
    pub terrain: Terrain,
    pub seed: u64,
    pub steps: u32,
    pipeline: PhysicsPipeline,
    gravity: Vector<Real>,
    integration: IntegrationParameters,
    islands: IslandManager,
    broad: BroadPhaseBvh,
    narrow: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    joints: ImpulseJointSet,
    multijoints: MultibodyJointSet,
    ccd: CCDSolver,
    vehicles: Vec<Vehicle>,
}
impl World {
    pub fn new(scenario: Scenario, seed: u64) -> Result<Self, String> {
        scenario.validate()?;
        let terrain = Terrain {
            kind: scenario.terrain,
            mixed: scenario.mixed_material,
        };
        let mut w = Self {
            integration: IntegrationParameters {
                dt: 1. / scenario.physics_hz as f32,
                ..Default::default()
            },
            scenario,
            terrain,
            seed,
            steps: 0,
            pipeline: PhysicsPipeline::new(),
            gravity: vector![0., -9.81, 0.],
            islands: IslandManager::new(),
            broad: BroadPhaseBvh::new(),
            narrow: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            joints: ImpulseJointSet::new(),
            multijoints: MultibodyJointSet::new(),
            ccd: CCDSolver::new(),
            vehicles: Vec::new(),
        };
        w.terrain.insert(&mut w.colliders);
        let mut rng = seed;
        for cfg in &w.scenario.vehicles {
            // SplitMix64: reset determinism does not depend on a third-party RNG version.
            rng = rng.wrapping_add(0x9e3779b97f4a7c15);
            let mut r = rng;
            r = (r ^ (r >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            r = (r ^ (r >> 27)).wrapping_mul(0x94d049bb133111eb);
            r ^= r >> 31;
            let jitter = ((r >> 40) as f32 / 16777216. - 0.5) * 2. * w.scenario.spawn_jitter;
            let x = cfg.spawn[0] + jitter;
            let z = cfg.spawn[1];
            let s = cfg.spec();
            let body = w.bodies.insert(
                RigidBodyBuilder::dynamic()
                    .translation(vector![x, w.terrain.height(x, z) + 0.85, z])
                    .linear_damping(0.08)
                    .angular_damping(0.8)
                    .ccd_enabled(true),
            );
            let collider = w.colliders.insert_with_parent(
                ColliderBuilder::cuboid(s.width / 2., 0.25, s.length / 2.)
                    .mass(s.mass)
                    .friction(0.4)
                    .user_data(1000 + w.vehicles.len() as u128),
                body,
                &mut w.bodies,
            );
            let mut dynamics = DynamicRayCastVehicleController::new(body);
            dynamics.index_up_axis = 1;
            dynamics.index_forward_axis = 2;
            let tuning = WheelTuning {
                suspension_stiffness: 35.,
                suspension_compression: 4.4,
                suspension_damping: 5.2,
                max_suspension_travel: 0.25,
                max_suspension_force: 20000.,
                friction_slip: 1.1,
                ..Default::default()
            };
            for wz in [s.wheelbase / 2., -s.wheelbase / 2.] {
                for wx in [-s.width * 0.48, s.width * 0.48] {
                    dynamics.add_wheel(
                        point![wx, -0.1, wz],
                        -Vector::y(),
                        -Vector::x(),
                        0.45,
                        0.32,
                        &tuning,
                    );
                }
            }
            w.vehicles.push(Vehicle {
                body,
                collider,
                dynamics,
                action: Action::default(),
                terminated: false,
                truncated: false,
                reason: String::new(),
                materials: vec![0; 4],
            });
        }
        // Settle suspension and initialize broad-phase queries before the first observation.
        for _ in 0..60 {
            w.substep(&vec![Action::default(); w.vehicles.len()]);
        }
        Ok(w)
    }
    pub fn reset(&mut self, seed: u64) -> Result<StepResult, String> {
        *self = Self::new(self.scenario.clone(), seed)?;
        Ok(self.result(vec![0.; self.vehicles.len()]))
    }
    pub fn active(&self) -> Vec<bool> {
        self.vehicles
            .iter()
            .map(|v| !v.terminated && !v.truncated)
            .collect()
    }
    pub fn initial(&self) -> StepResult {
        self.result(vec![0.; self.vehicles.len()])
    }
    pub fn validate_actions(&self, actions: &[Action]) -> Result<(), String> {
        if actions.len() != self.vehicles.len() {
            return Err("one action per stable vehicle slot is required".into());
        }
        for a in actions {
            a.validate()?;
        }
        Ok(())
    }
    pub fn step(&mut self, actions: &[Action]) -> Result<StepResult, String> {
        self.validate_actions(actions)?;
        if !self.active().iter().any(|v| *v) {
            return Err("episode finished; reset before stepping".into());
        }
        let before: Vec<f32> = self
            .vehicles
            .iter()
            .map(|v| self.bodies[v.body].translation().z)
            .collect();
        let active = self.active();
        let mut collisions = vec![false; self.vehicles.len()];
        for _ in 0..self.scenario.physics_hz / self.scenario.control_hz {
            self.substep(actions);
            // A contact may begin and end between policy ticks. Latch every substep.
            for (v, collided) in self.vehicles.iter().zip(&mut collisions) {
                *collided |= self.narrow.contact_pairs_with(v.collider).any(|pair| {
                    let other = if pair.collider1 == v.collider {
                        pair.collider2
                    } else {
                        pair.collider1
                    };
                    pair.has_any_active_contact && self.colliders[other].user_data >= 100
                });
            }
        }
        self.steps += 1;
        let mut rewards = vec![0.; self.vehicles.len()];
        for (i, v) in self.vehicles.iter_mut().enumerate() {
            if !active[i] {
                continue;
            }
            v.action = actions[i];
            let b = &self.bodies[v.body];
            let p = b.translation();
            let collision = collisions[i];
            let offroad = p.x.abs() > 28.
                || p.z < -12.
                || p.z > 108.
                || (b.rotation() * Vector::y()).y < 0.35
                || p.y < -5.;
            let success = p.z >= 90.;
            rewards[i] = (p.z - before[i])
                - 0.002 * actions[i].steering.powi(2)
                - 0.005 * (p.x - route_x(p.z)).abs();
            if collision || offroad || success {
                v.terminated = true;
                v.reason = if collision {
                    "collision"
                } else if offroad {
                    "offroad"
                } else {
                    "success"
                }
                .into();
                rewards[i] += if success && !collision && !offroad {
                    20.
                } else {
                    -10.
                };
            } else if self.steps >= self.scenario.max_steps {
                v.truncated = true;
                v.reason = "time_limit".into();
            }
        }
        Ok(self.result(rewards))
    }
    fn substep(&mut self, actions: &[Action]) {
        for (i, v) in self.vehicles.iter_mut().enumerate() {
            let done = v.terminated || v.truncated;
            let a = if done {
                Action {
                    throttle: -1.,
                    steering: 0.,
                }
            } else {
                actions[i]
            };
            let spec = self.scenario.vehicles[i].spec();
            // Query the actual wheel ray for THIS substep; never infer a bridge material from X/Z alone.
            let b = &self.bodies[v.body];
            let queries = self.broad.as_query_pipeline(
                self.narrow.query_dispatcher(),
                &self.bodies,
                &self.colliders,
                QueryFilter::default().exclude_rigid_body(v.body),
            );
            let speed = b.linvel().dot(&(b.rotation() * Vector::z()));
            for (j, wheel) in v.dynamics.wheels_mut().iter_mut().enumerate() {
                let origin = b.position() * wheel.chassis_connection_point_cs;
                let dir = b.rotation() * wheel.direction_cs;
                let hit = queries.cast_ray(
                    &Ray::new(origin, dir),
                    wheel.suspension_rest_length + wheel.radius,
                    true,
                );
                let material = hit
                    .map(|(handle, t)| {
                        let p = origin + dir * t;
                        if self.colliders[handle].user_data < 100 {
                            self.terrain.material(p.x, p.z)
                        } else {
                            Material {
                                id: 3,
                                grip: 0.8,
                                rolling: 0.02,
                            }
                        }
                    })
                    .unwrap_or(Material {
                        id: 0,
                        grip: 0.,
                        rolling: 0.,
                    });
                v.materials[j] = material.id;
                wheel.friction_slip = material.grip;
                wheel.steering = if j < 2 {
                    a.steering * spec.steering
                } else {
                    0.
                };
                wheel.engine_force = a.throttle.max(0.)
                    * (spec.engine / 4.).min(material.grip * spec.mass * 9.81 / 4.);
                wheel.brake = (-a.throttle).max(0.) * spec.mass * 0.025
                    + material.rolling * spec.mass * 9.81 * self.integration.dt / 4.
                        * speed.abs().min(1.);
            }
            let queries = self.broad.as_query_pipeline_mut(
                self.narrow.query_dispatcher(),
                &mut self.bodies,
                &mut self.colliders,
                QueryFilter::default().exclude_rigid_body(v.body),
            );
            v.dynamics.update_vehicle(self.integration.dt, queries);
        }
        self.pipeline.step(
            &self.gravity,
            &self.integration,
            &mut self.islands,
            &mut self.broad,
            &mut self.narrow,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.joints,
            &mut self.multijoints,
            &mut self.ccd,
            &(),
            &(),
        );
    }
    pub fn observations(&self) -> Vec<Observation> {
        self.vehicles
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut o = [0.; OBS_DIM];
                let b = &self.bodies[v.body];
                let p = b.translation();
                let vel = b.rotation().inverse() * b.linvel();
                let target_z = (p.z + 8.).min(94.);
                let target =
                    b.rotation().inverse() * (vector![route_x(target_z), p.y, target_z] - p);
                o[0] = vel.z / 10.;
                o[1] = vel.x / 10.;
                o[2] = b.angvel().y;
                o[3] = target.x / 10.;
                o[4] = target.z / 10.;
                o[5] = (90. - p.z) / 100.;
                o[6] = v.action.throttle;
                o[7] = v.action.steering;
                let cfg = &self.scenario.vehicles[i];
                let predicate = |h: ColliderHandle, _: &Collider| {
                    cfg.aware || !self.vehicles.iter().any(|v| v.collider == h)
                };
                let queries = self.broad.as_query_pipeline(
                    self.narrow.query_dispatcher(),
                    &self.bodies,
                    &self.colliders,
                    QueryFilter::default()
                        .exclude_rigid_body(v.body)
                        .predicate(&predicate),
                );
                for (k, angle) in [-0.8_f32, -0.4, 0., 0.4, 0.8].iter().enumerate() {
                    let dir = b.rotation() * vector![angle.sin(), 0., angle.cos()];
                    let ray = Ray::new(Point::from(*p), dir);
                    o[8 + k] = queries
                        .cast_ray(&ray, cfg.sensing_range, true)
                        .map(|(_, t)| t / cfg.sensing_range)
                        .unwrap_or(1.);
                }
                if cfg.aware {
                    let mut best = cfg.sensing_range;
                    for (j, other) in self.vehicles.iter().enumerate() {
                        if i == j {
                            continue;
                        }
                        let ob = &self.bodies[other.body];
                        let delta = ob.translation() - p;
                        let distance = delta.norm();
                        let local = b.rotation().inverse() * delta;
                        if distance >= best || local.z < 0. {
                            continue;
                        }
                        let ray = Ray::new(Point::from(*p), delta.normalize());
                        if queries
                            .cast_ray(&ray, distance, true)
                            .is_some_and(|(h, _)| h == other.collider)
                        {
                            best = distance;
                            let dv = b.rotation().inverse() * (ob.linvel() - b.linvel());
                            o[13] = local.x / cfg.sensing_range;
                            o[14] = local.z / cfg.sensing_range;
                            o[15] = dv.x / 10.;
                            o[16] = dv.z / 10.;
                            o[17] = 1.;
                        }
                    }
                }
                o
            })
            .collect()
    }
    fn result(&self, rewards: Vec<f32>) -> StepResult {
        StepResult {
            observations: self.observations(),
            rewards,
            terminated: self.vehicles.iter().map(|v| v.terminated).collect(),
            truncated: self.vehicles.iter().map(|v| v.truncated).collect(),
            active: self.active(),
            reasons: self.vehicles.iter().map(|v| v.reason.clone()).collect(),
        }
    }
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            time: self.steps as f32 / self.scenario.control_hz as f32,
            vehicles: self
                .vehicles
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let b = &self.bodies[v.body];
                    let q = b.rotation().quaternion();
                    VehicleSnapshot {
                        id: self.scenario.vehicles[i].id.clone(),
                        position: (*b.translation()).into(),
                        rotation: [q.i, q.j, q.k, q.w],
                        speed: b.linvel().norm(),
                        wheels: v
                            .dynamics
                            .wheels()
                            .iter()
                            .map(|w| w.center().into())
                            .collect(),
                        materials: v.materials.clone(),
                        active: !v.terminated && !v.truncated,
                    }
                })
                .collect(),
        }
    }
}
pub fn route_x(z: f32) -> f32 {
    2. * (z * 0.045).sin()
}
