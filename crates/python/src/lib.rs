use ml_cars_controllers::{Batch, Policy};
use ml_cars_sim::{Action, OBS_DIM, Scenario, StepResult};
use numpy::{
    IntoPyArray, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
    ndarray::{Array2, Array3},
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}
fn pack<'py>(py: Python<'py>, results: Vec<StepResult>) -> PyResult<Bound<'py, PyDict>> {
    let b = results.len();
    let n = results[0].observations.len();
    let d = PyDict::new(py);
    let obs: Vec<f32> = results
        .iter()
        .flat_map(|r| r.observations.iter().flatten().copied())
        .collect();
    d.set_item(
        "observations",
        Array3::from_shape_vec((b, n, OBS_DIM), obs)
            .unwrap()
            .into_pyarray(py),
    )?;
    d.set_item(
        "rewards",
        Array2::from_shape_vec(
            (b, n),
            results
                .iter()
                .flat_map(|r| r.rewards.iter().copied())
                .collect(),
        )
        .unwrap()
        .into_pyarray(py),
    )?;
    for name in ["terminated", "truncated", "active"] {
        let values: Vec<bool> = results
            .iter()
            .flat_map(|r| {
                match name {
                    "terminated" => &r.terminated,
                    "truncated" => &r.truncated,
                    _ => &r.active,
                }
                .iter()
                .copied()
            })
            .collect();
        d.set_item(
            name,
            Array2::from_shape_vec((b, n), values)
                .unwrap()
                .into_pyarray(py),
        )?;
    }
    d.set_item(
        "reasons",
        results.into_iter().map(|r| r.reasons).collect::<Vec<_>>(),
    )?;
    Ok(d)
}
#[pyclass(name = "Batch")]
struct PyBatch {
    batch: Batch,
}
#[pymethods]
impl PyBatch {
    #[new]
    fn new(py: Python<'_>, scenario_json: &str, seeds: Vec<u64>) -> PyResult<Self> {
        let scenario = Scenario::from_json(scenario_json).map_err(err)?;
        Ok(Self {
            batch: py
                .detach(move || Batch::new(scenario, &seeds))
                .map_err(err)?,
        })
    }
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pack(
            py,
            self.batch
                .worlds
                .iter()
                .map(|w| w.world.initial())
                .collect(),
        )
    }
    fn reset<'py>(&mut self, py: Python<'py>, seeds: Vec<u64>) -> PyResult<Bound<'py, PyDict>> {
        if seeds.len() != self.batch.worlds.len() {
            return Err(err("one seed per world required".into()));
        }
        let results = py
            .detach(|| {
                self.batch
                    .worlds
                    .iter_mut()
                    .zip(seeds)
                    .map(|(w, s)| w.reset(s))
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(err)?;
        pack(py, results)
    }
    fn reset_at<'py>(
        &mut self,
        py: Python<'py>,
        index: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let w = self
            .batch
            .worlds
            .get_mut(index)
            .ok_or_else(|| err("world index out of bounds".into()))?;
        let result = py.detach(|| w.reset(seed)).map_err(err)?;
        pack(py, vec![result])
    }
    #[pyo3(signature=(actions, override_native=false))]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray3<'py, f32>,
        override_native: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let b = self.batch.worlds.len();
        let n = self.batch.worlds[0].world.scenario.vehicles.len();
        if actions.shape() != [b, n, 2] {
            return Err(err(format!(
                "expected float32 actions of shape ({b}, {n}, 2)"
            )));
        }
        let values = actions
            .as_slice()
            .map_err(|_| err("actions must be C-contiguous".into()))?;
        let mut all = Vec::new();
        for (i, world_values) in values.chunks(n * 2).enumerate() {
            let mut row = Vec::new();
            for (j, v) in world_values.chunks(2).enumerate() {
                let a = Action {
                    throttle: v[0],
                    steering: v[1],
                };
                a.validate().map_err(err)?;
                row.push(
                    if override_native
                        || self.batch.worlds[i].world.scenario.vehicles[j].controller == "external"
                    {
                        Some(a)
                    } else {
                        None
                    },
                );
            }
            all.push(row);
        }
        let results = py.detach(|| self.batch.step(&all)).map_err(err)?;
        pack(py, results)
    }
    fn step_native<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let actions = self
            .batch
            .worlds
            .iter()
            .map(|w| vec![None; w.world.scenario.vehicles.len()])
            .collect::<Vec<_>>();
        let results = py.detach(|| self.batch.step(&actions)).map_err(err)?;
        pack(py, results)
    }
    fn snapshot_json(&self, index: usize) -> PyResult<String> {
        let w = self
            .batch
            .worlds
            .get(index)
            .ok_or_else(|| err("world index out of bounds".into()))?;
        serde_json::to_string(&w.world.snapshot()).map_err(|e| err(e.to_string()))
    }
}
#[pyfunction]
fn policy_actions<'py>(
    py: Python<'py>,
    path: &str,
    observations: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    if observations.shape()[1] != OBS_DIM {
        return Err(err("observation dimension mismatch".into()));
    }
    let p = Policy::load(path).map_err(err)?;
    let mut out = Vec::new();
    for row in observations.as_array().rows() {
        let o: Vec<f32> = row.iter().copied().collect();
        if o.iter().any(|x| !x.is_finite()) {
            return Err(err("observations must be finite".into()));
        }
        let a = p.infer(o.as_slice().try_into().unwrap());
        out.extend([a.throttle, a.steering]);
    }
    Ok(Array2::from_shape_vec((observations.shape()[0], 2), out)
        .unwrap()
        .into_pyarray(py))
}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBatch>()?;
    m.add_function(wrap_pyfunction!(policy_actions, m)?)?;
    m.add("OBS_DIM", OBS_DIM)?;
    Ok(())
}
