use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VehicleConfig {
    pub id: String,
    pub kind: VehicleKind,
    pub controller: String,
    pub aware: bool,
    pub sensing_range: f32,
    pub spawn: [f32; 2],
    pub parameters: Option<VehicleSpec>,
}
impl Default for VehicleConfig {
    fn default() -> Self {
        Self {
            id: "car_0".into(),
            kind: VehicleKind::Car,
            controller: "native".into(),
            aware: true,
            sensing_range: 30.,
            spawn: [0., 0.],
            parameters: None,
        }
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VehicleKind {
    Car,
    Truck,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VehicleSpec {
    pub mass: f32,
    pub width: f32,
    pub length: f32,
    pub wheelbase: f32,
    pub steering: f32,
    pub engine: f32,
}
impl Default for VehicleSpec {
    fn default() -> Self {
        VehicleKind::Car.spec()
    }
}
impl VehicleConfig {
    pub fn spec(&self) -> VehicleSpec {
        self.parameters.unwrap_or_else(|| self.kind.spec())
    }
}
impl VehicleKind {
    pub fn spec(self) -> VehicleSpec {
        match self {
            Self::Car => VehicleSpec {
                mass: 900.,
                width: 1.7,
                length: 3.4,
                wheelbase: 2.4,
                steering: 0.55,
                engine: 4200.,
            },
            Self::Truck => VehicleSpec {
                mass: 2200.,
                width: 2.1,
                length: 4.8,
                wheelbase: 3.5,
                steering: 0.38,
                engine: 6500.,
            },
        }
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TerrainKind {
    Flat,
    Hills,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Scenario {
    pub version: u32,
    pub terrain: TerrainKind,
    pub mixed_material: bool,
    pub physics_hz: u32,
    pub control_hz: u32,
    pub max_steps: u32,
    pub spawn_jitter: f32,
    pub vehicles: Vec<VehicleConfig>,
}
impl Default for Scenario {
    fn default() -> Self {
        Self {
            version: 1,
            terrain: TerrainKind::Flat,
            mixed_material: true,
            physics_hz: 120,
            control_hz: 20,
            max_steps: 800,
            spawn_jitter: 0.25,
            vehicles: vec![VehicleConfig::default()],
        }
    }
}
impl Scenario {
    pub fn from_json(s: &str) -> Result<Self, String> {
        let v: Self = serde_json::from_str(s).map_err(|e| e.to_string())?;
        v.validate()?;
        Ok(v)
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.version != 1 {
            return Err("unsupported scenario version".into());
        }
        if self.control_hz == 0
            || self.physics_hz < self.control_hz
            || !self.physics_hz.is_multiple_of(self.control_hz)
            || self.physics_hz > 1000
        {
            return Err("physics_hz must be a multiple of control_hz and at most 1000".into());
        }
        if self.max_steps == 0 || self.vehicles.is_empty() || self.vehicles.len() > 64 {
            return Err("require max_steps > 0 and 1..=64 vehicles".into());
        }
        if !self.spawn_jitter.is_finite() || !(0.0..=2.).contains(&self.spawn_jitter) {
            return Err("spawn_jitter must be finite and in [0,2]".into());
        }
        let mut ids = std::collections::HashSet::new();
        for v in &self.vehicles {
            let s = v.spec();
            if [s.mass, s.width, s.length, s.wheelbase, s.steering, s.engine]
                .iter()
                .any(|x| !x.is_finite())
                || !(200.0..=10000.).contains(&s.mass)
                || !(1.0..=4.).contains(&s.width)
                || !(2.0..=10.).contains(&s.length)
                || !(1.0..=s.length).contains(&s.wheelbase)
                || !(0.1..=0.8).contains(&s.steering)
                || !(100.0..=30000.).contains(&s.engine)
            {
                return Err("invalid vehicle parameters".into());
            }
            if v.id.is_empty() || !ids.insert(&v.id) {
                return Err("vehicle IDs must be nonempty and unique".into());
            }
            if !v.sensing_range.is_finite() || !(1.0..=100.).contains(&v.sensing_range) {
                return Err("sensing_range must be in [1,100]".into());
            }
            if !v.spawn.iter().all(|n| n.is_finite())
                || v.spawn[0].abs() > 24.
                || !(-10.0..=85.).contains(&v.spawn[1])
            {
                return Err("spawn outside safe terrain bounds".into());
            }
            if v.controller != "native"
                && v.controller != "external"
                && !v.controller.starts_with("policy:")
            {
                return Err("controller must be native, external, or policy:<path>".into());
            }
        }
        Ok(())
    }
}
