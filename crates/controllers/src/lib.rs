mod onnx;
use ml_cars_sim::{Action, OBS_DIM, Observation, Scenario, Snapshot, StepResult, World};
pub use onnx::OnnxPolicy;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{fs, sync::Arc};
pub trait Controller: Send + Sync {
    fn action(&mut self, observation: &Observation) -> Result<Action, String>;
    fn reset(&mut self);
}
pub struct PathFollower {
    integral: f32,
    dt: f32,
}
impl Default for PathFollower {
    fn default() -> Self {
        Self::new(0.05)
    }
}
impl PathFollower {
    pub fn new(dt: f32) -> Self {
        Self { integral: 0., dt }
    }
}
impl Controller for PathFollower {
    fn action(&mut self, o: &Observation) -> Result<Action, String> {
        let target = if o[17] > 0.5 && o[14] < 0.35 && o[13].abs() < 0.1 {
            0.
        } else {
            5.
        };
        let error = target - o[0] * 10.;
        self.integral = (self.integral + error * self.dt).clamp(-1., 1.);
        Ok(Action {
            throttle: (error * 0.5 + self.integral * 0.1).clamp(-1., 1.),
            steering: (o[3].atan2(o[4]) * 1.8 - o[2] * 0.15).clamp(-1., 1.),
        })
    }
    fn reset(&mut self) {
        self.integral = 0.;
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub version: u32,
    pub observation_schema: String,
    pub action_schema: String,
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
    pub layers: Vec<Layer>,
}
impl Policy {
    pub fn load(path: &str) -> Result<Self, String> {
        let p: Self =
            serde_json::from_str(&fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)
                .map_err(|e| e.to_string())?;
        p.validate()?;
        Ok(p)
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.version != 1
            || self.observation_schema != "ml_cars/obs-v1"
            || self.action_schema != "throttle-steering/tanh-v1"
        {
            return Err("unsupported policy schema".into());
        }
        if self.mean.len() != OBS_DIM
            || self.scale.len() != OBS_DIM
            || self.mean.iter().any(|x| !x.is_finite())
            || self.scale.iter().any(|x| !x.is_finite() || *x <= 0.)
            || self.layers.is_empty()
            || self.layers.len() > 8
        {
            return Err("invalid policy normalization or layers".into());
        }
        let mut width = OBS_DIM;
        for l in &self.layers {
            if l.bias.is_empty()
                || l.bias.len() > 4096
                || l.weights.len() != l.bias.len()
                || l.weights
                    .iter()
                    .any(|r| r.len() != width || r.iter().any(|x| !x.is_finite()))
                || l.bias.iter().any(|x| !x.is_finite())
            {
                return Err("invalid policy tensor shape or nonfinite weights".into());
            }
            width = l.bias.len();
        }
        if width != 2 {
            return Err("policy must output two actions".into());
        }
        Ok(())
    }
    pub fn infer(&self, o: &Observation) -> Action {
        let mut x: Vec<f32> = o
            .iter()
            .zip(&self.mean)
            .zip(&self.scale)
            .map(|((x, m), s)| (x - m) / s)
            .collect();
        for l in &self.layers {
            x = l
                .weights
                .iter()
                .zip(&l.bias)
                .map(|(w, b)| (w.iter().zip(&x).fold(*b, |s, (w, x)| s + w * x)).tanh())
                .collect();
        }
        Action {
            throttle: x[0],
            steering: x[1],
        }
    }
}
pub struct NeuralController {
    pub policy: Arc<Policy>,
}
impl Controller for NeuralController {
    fn action(&mut self, o: &Observation) -> Result<Action, String> {
        Ok(self.policy.infer(o))
    }
    fn reset(&mut self) {}
}
/// Load a feedforward JSON or ONNX policy through the same fallible interface.
pub fn load_policy(path: &str) -> Result<Box<dyn Controller>, String> {
    if is_onnx(path) {
        Ok(Box::new(OnnxPolicy::load(path)?))
    } else {
        Ok(Box::new(NeuralController {
            policy: Arc::new(Policy::load(path)?),
        }))
    }
}
fn is_onnx(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
}
pub struct Runner {
    pub world: World,
    controllers: Vec<Option<Box<dyn Controller>>>,
}
impl Runner {
    pub fn new(scenario: Scenario, seed: u64) -> Result<Self, String> {
        scenario.validate()?;
        let mut policies = std::collections::HashMap::new();
        let controllers = scenario
            .vehicles
            .iter()
            .map(|v| {
                Ok(match v.controller.as_str() {
                    "external" => None,
                    "native" => Some(Box::new(PathFollower::new(1. / scenario.control_hz as f32))
                        as Box<dyn Controller>),
                    s => {
                        let path = s.strip_prefix("policy:").ok_or("invalid controller")?;
                        if is_onnx(path) {
                            return load_policy(path).map(Some);
                        }
                        let policy = if let Some(p) = policies.get(path) {
                            Arc::clone(p)
                        } else {
                            let p = Arc::new(Policy::load(path)?);
                            policies.insert(path.to_owned(), Arc::clone(&p));
                            p
                        };
                        Some(Box::new(NeuralController { policy }) as Box<dyn Controller>)
                    }
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            world: World::new(scenario, seed)?,
            controllers,
        })
    }
    pub fn validate(&self, external: &[Option<Action>]) -> Result<(), String> {
        if external.len() != self.controllers.len() {
            return Err("action slots must match vehicle count".into());
        }
        let active = self.world.active();
        if !active.iter().any(|x| *x) {
            return Err("episode finished; reset required".into());
        }
        for (i, a) in external.iter().enumerate() {
            if let Some(a) = a {
                a.validate()?;
            } else if active[i] && self.controllers[i].is_none() {
                return Err(format!(
                    "missing action for {}",
                    self.world.scenario.vehicles[i].id
                ));
            }
        }
        Ok(())
    }
    pub fn step(&mut self, external: &[Option<Action>]) -> Result<StepResult, String> {
        let actions = self.prepare_actions(external)?;
        self.world.step(&actions)
    }
    fn prepare_actions(&mut self, external: &[Option<Action>]) -> Result<Vec<Action>, String> {
        self.validate(external)?;
        let obs = self.world.observations();
        let active = self.world.active();
        let actions: Vec<_> = self
            .controllers
            .iter_mut()
            .enumerate()
            .map(|(i, c)| {
                if !active[i] {
                    Ok(Action::default())
                } else if let Some(action) = external[i] {
                    Ok(action)
                } else {
                    c.as_mut()
                        .expect("validated controller")
                        .action(&obs[i])
                        .map_err(|e| {
                            format!("controller {}: {e}", self.world.scenario.vehicles[i].id)
                        })
                }
            })
            .collect::<Result<_, String>>()?;
        self.world.validate_actions(&actions)?;
        Ok(actions)
    }
    pub fn reset(&mut self, seed: u64) -> Result<StepResult, String> {
        let result = self.world.reset(seed)?;
        for c in self.controllers.iter_mut().flatten() {
            c.reset();
        }
        Ok(result)
    }
}
pub struct Batch {
    pub worlds: Vec<Runner>,
}
impl Batch {
    pub fn new(scenario: Scenario, seeds: &[u64]) -> Result<Self, String> {
        if seeds.is_empty() {
            return Err("batch must not be empty".into());
        }
        Ok(Self {
            worlds: seeds
                .par_iter()
                .map(|s| Runner::new(scenario.clone(), *s))
                .collect::<Result<_, _>>()?,
        })
    }
    pub fn step(&mut self, actions: &[Vec<Option<Action>>]) -> Result<Vec<StepResult>, String> {
        if actions.len() != self.worlds.len() {
            return Err("batch action count mismatch".into());
        }
        // Validate the whole batch before mutating any world or controller state.
        for (w, a) in self.worlds.iter().zip(actions) {
            w.validate(a)?;
        }
        // Runtime inference is fallible too. Collect every world's actions
        // before advancing physics, so an error cannot leave a partial batch.
        let prepared = self
            .worlds
            .par_iter_mut()
            .zip(actions)
            .map(|(w, a)| w.prepare_actions(a))
            .collect::<Result<Vec<_>, _>>()?;
        self.worlds
            .par_iter_mut()
            .zip(&prepared)
            .map(|(w, a)| w.world.step(a))
            .collect()
    }
}
#[derive(Serialize, Deserialize)]
pub struct Recording {
    pub version: u32,
    pub scenario: Scenario,
    pub seed: u64,
    pub frames: Vec<Snapshot>,
}
impl Recording {
    pub fn load(path: &str) -> Result<Self, String> {
        let r: Self = serde_json::from_str(&fs::read_to_string(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        r.scenario.validate()?;
        if r.version != 1
            || r.frames.is_empty()
            || r.frames.iter().any(|f| {
                f.vehicles.len() != r.scenario.vehicles.len()
                    || !f.time.is_finite()
                    || f.vehicles
                        .iter()
                        .any(|v| v.position.iter().chain(&v.rotation).any(|x| !x.is_finite()))
            })
        {
            return Err("invalid recording".into());
        }
        Ok(r)
    }
    pub fn save(&self, path: &str) -> Result<(), String> {
        fs::write(path, serde_json::to_vec(self).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod inference_failures {
    use super::*;
    struct FailingPolicy {
        nonfinite: bool,
    }
    impl Controller for FailingPolicy {
        fn action(&mut self, _: &Observation) -> Result<Action, String> {
            if self.nonfinite {
                Ok(Action {
                    throttle: f32::NAN,
                    steering: 0.,
                })
            } else {
                Err("test inference failure".into())
            }
        }
        fn reset(&mut self) {}
    }
    #[test]
    fn inference_errors_do_not_partially_advance_batch_physics() {
        for nonfinite in [false, true] {
            let mut batch = Batch::new(Scenario::default(), &[1, 2]).unwrap();
            batch.worlds[1].controllers[0] = Some(Box::new(FailingPolicy { nonfinite }));
            let before: Vec<_> = batch
                .worlds
                .iter()
                .map(|w| w.world.observations())
                .collect();
            assert!(batch.step(&[vec![None], vec![None]]).is_err());
            for (world, observations) in batch.worlds.iter().zip(before) {
                assert_eq!(world.world.steps, 0);
                assert_eq!(world.world.observations(), observations);
            }
        }
    }
}
