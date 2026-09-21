//! CPU ONNX inference. Normalization belongs to the graph, never to a sidecar.
use crate::Controller;
use ml_cars_sim::{Action, OBS_DIM, Observation};
use ort::{
    session::Session,
    value::{TensorElementType, TensorRef, ValueType},
};
use std::{
    env,
    path::{Path, PathBuf},
};

pub struct OnnxPolicy {
    session: Session,
}

impl OnnxPolicy {
    pub fn load(path: &str) -> Result<Self, String> {
        if !Path::new(path).is_file() {
            return Err(format!("ONNX model not found: {path}"));
        }
        let library = env::var_os("ORT_DYLIB_PATH")
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(if cfg!(target_os = "windows") {
                    "onnxruntime.dll"
                } else if cfg!(target_os = "macos") {
                    "libonnxruntime.dylib"
                } else {
                    "libonnxruntime.so"
                })
            });
        // Use the fallible loader before any ort API which would otherwise panic
        // on an absent or incompatible runtime. No Python is involved in loading.
        ort::init_from(&library).map_err(|e| format!("ONNX Runtime unavailable: {e}. Set ORT_DYLIB_PATH to a CPU ONNX Runtime >= 1.22 shared library; see README.md."))?
            .with_name("ml_cars").with_telemetry(false).commit();
        let session = Session::builder()
            .map_err(runtime_error)?
            .with_intra_threads(1)
            .map_err(runtime_error)?
            .with_inter_threads(1)
            .map_err(runtime_error)?
            .commit_from_file(path)
            .map_err(|e| format!("cannot load ONNX policy {path}: {e}"))?;
        // Each vehicle owns its session; Rayon parallelizes worlds. Avoid an
        // additional native thread pool per car and shared mutable sessions.
        if session.inputs().len() != 1 || session.outputs().len() != 1 {
            return Err("ONNX policy requires exactly one input and one output; recurrent policies are unsupported".into());
        }
        let input = &session.inputs()[0];
        let output = &session.outputs()[0];
        if input.name() != "observations"
            || output.name() != "actions"
            || !valid_tensor(input.dtype(), OBS_DIM as i64)
            || !valid_tensor(output.dtype(), 2)
        {
            return Err("ONNX policy requires float32 observations[1,18] -> actions[1,2] (a dynamic batch dimension is also accepted)".into());
        }
        let metadata = session.metadata().map_err(runtime_error)?;
        for (key, expected) in [
            ("ml_cars.version", "1"),
            ("ml_cars.observation_schema", "ml_cars/obs-v1"),
            ("ml_cars.action_schema", "throttle-steering/tanh-v1"),
            ("ml_cars.normalization", "embedded/v1"),
            ("ml_cars.recurrent", "false"),
        ] {
            if metadata.custom(key).as_deref() != Some(expected) {
                return Err(format!("ONNX metadata {key} must equal {expected:?}"));
            }
        }
        drop(metadata);
        Ok(Self { session })
    }
    pub fn infer(&mut self, observation: &Observation) -> Result<Action, String> {
        if observation.iter().any(|x| !x.is_finite()) {
            return Err("ONNX observations must be finite".into());
        }
        let input = TensorRef::from_array_view(([1usize, OBS_DIM], observation.as_slice()))
            .map_err(runtime_error)?;
        let outputs = self
            .session
            .run(ort::inputs!["observations" => input])
            .map_err(runtime_error)?;
        let (shape, values) = outputs["actions"]
            .try_extract_tensor::<f32>()
            .map_err(runtime_error)?;
        if shape.as_ref() != [1, 2] || values.len() != 2 {
            return Err("ONNX runtime output must have shape [1,2]".into());
        }
        let action = Action {
            throttle: values[0],
            steering: values[1],
        };
        action
            .validate()
            .map_err(|e| format!("invalid ONNX output: {e}"))?;
        Ok(action)
    }
}
impl Controller for OnnxPolicy {
    fn action(&mut self, observation: &Observation) -> Result<Action, String> {
        self.infer(observation)
    }
    fn reset(&mut self) {} // Contract explicitly disallows recurrent state.
}
fn valid_tensor(value: &ValueType, width: i64) -> bool {
    value.tensor_type() == Some(TensorElementType::Float32)
        && value
            .tensor_shape()
            .is_some_and(|s| s.len() == 2 && (s[0] == 1 || s[0] == -1) && s[1] == width)
}
fn runtime_error(error: impl std::fmt::Display) -> String {
    format!("ONNX inference: {error}")
}
