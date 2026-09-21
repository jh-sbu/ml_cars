# ML Cars

A Rust vehicle training gym with batched Python environments, native policy
execution, and a Bevy 3D viewer. The MVP in [DESIGN.md](DESIGN.md) is implemented
with Rapier raycast wheels and feedforward policy deployment through JSON or ONNX.

Run commands from the repository root. Rust dependencies are locked in
`Cargo.lock`; Rust 1.89+ and Python 3.10+ are required. The viewer needs a graphics
driver and a desktop session (Linux builds enable X11).

```sh
# Native baseline, no Python or window required
cargo run --release -- run scenarios/flat.json
cargo run --release -- run scenarios/hills.json 42 recordings.json

# Included trained policy, entirely in Rust
cargo run --release -- run scenarios/learned.json
cargo run --release -- run scenarios/mixed.json

# Live evaluation and pose-recording playback
cargo run -p ml-cars-viewer -- scenarios/learned.json
cargo run -p ml-cars-viewer -- --replay recordings.json
```

Viewer controls: **Space** pauses/resumes, **N** advances one policy step,
**R** resets the seed or rewinds playback, and **Tab** cycles vehicle-follow
cameras. Cyan vehicles have awareness; amber vehicles do not. The overlay shows
speed, wheel material IDs, and lifecycle state. Live range rays and the route
are drawn over the shared collision terrain. Rendering advances an accumulator
at the scenario control frequency; it never owns a second physics world.

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install maturin
maturin develop
uv pip install 'torch>=2.6' pytest
python -m pytest -q
python -m ml_cars.train --output models/baseline.json
```

For a CPU-only PyTorch installation, use the official CPU wheel index:
`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`.
`python/requirements-validated.txt` records the Python dependency versions used
for validation; it includes the CPU PyTorch version.

The included baseline was trained using behavior cloning from the native route
follower. The training command collects demonstrations on both terrains, trains
an 18–32–32–2 Linear/Tanh network, exports its normalization and weights, checks
512 observations against Rust inference with absolute tolerance `2e-5`, checks
a standalone Rust process, and evaluates eight held-out episodes. The adjacent
[report](models/baseline.report.json) records seeds, versions, hashes, parity,
and evaluation results. This is a working training/deployment baseline, not a
claim of general driving competence or collision avoidance learned from traffic.

Python usage:

```python
import numpy as np
from ml_cars import DrivingEnv, ParallelDrivingEnv, Batch

env = DrivingEnv()  # one externally controlled vehicle
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(
    np.array([0.5, 0.0], dtype=np.float32)
)

# Background vehicles may have "native" or "policy:<path>" controllers.
# ParallelDrivingEnv exposes only vehicles assigned "external".
parallel = ParallelDrivingEnv({"vehicles": [
    {"id": "car", "controller": "external"},
    {"id": "truck", "kind": "truck", "controller": "external", "spawn": [4, 10]},
]})
observations, infos = parallel.reset(seed=42)
result = parallel.step({a: np.zeros(2, dtype=np.float32) for a in parallel.agents})

batch = Batch('{"vehicles":[{"id":"car","controller":"external"}]}', [1, 2, 3])
result = batch.step(np.zeros((3, 1, 2), dtype=np.float32))
# result: observations[B,N,18], rewards/terminated/truncated/active[B,N], reasons[B][N]
```

Each batch has a common scenario and stable vehicle ordering. Actions must be
C-contiguous float32 `[world, vehicle, 2]`. Native and learned slots ignore the
provided values unless `override_native=True`; all values must still be finite
and in range. `step_native()` needs no arrays when every active vehicle has an
internal controller. Rust releases the GIL during stepping and uses Rayon across
worlds; control its worker count with `RAYON_NUM_THREADS`. ONNX inference uses one
CPU thread per vehicle session. Returned arrays own
their memory and remain valid after subsequent steps.

There is **no automatic reset**. Inspect the final result, then use
`reset_at(index, seed)` for each completed world, or `reset(seeds)` for the batch.
A batch containing a completed world rejects stepping until that slot is reset.
Missing actions, invalid shapes/values, and completed-world errors are checked
before any world/controller advances. Finished agents leave PettingZoo's active
agent list; their final observations remain in the final step result and info.
Their physical chassis remains with braking applied and can still be hit or
slide on slopes. Controller inference failures return errors before any batch
physics advances; reset the batch before retrying, since native controller state
may have changed while preparing actions. Termination reasons are `success`, `collision`, and `offroad`;
`time_limit` is truncation. Gymnasium ends when its external vehicle finishes,
even if background vehicles remain active.

## Simulation contract

Units are meters, kilograms, seconds, and radians. Y is up, +Z is forward,
+X is right. A policy step collects all actions from the same observations,
then runs six physics steps by default (120 Hz physics / 20 Hz control).
`physics_hz` must be an integer multiple of `control_hz`. Resets settle the
suspension for 60 physics steps before setting episode time to zero.

Actions are `[throttle, steering]` in `[-1,1]`: positive throttle supplies
forward drive, negative throttle brakes, and positive steering turns right.
Reverse gear is not implemented. Per-wheel ray hits identify the collider before
material lookup. Grip limits drive force and tire friction; rolling resistance
adds braking impulse. Material IDs: 0 asphalt, 1 dirt, 2 ice, 3 other collider.
Both courses have an optional low-grip patch at `x >= 0, 30 <= z < 50`.
Terrain covers `x = [-32,32], z = [-16,112]` in two matching triangle-mesh tiles,
with 4 m samples. Heights use the exact collision triangle interpolation.
Hills change elevation and base traction. An obstacle stands at `(9, *, 45)`.

The route is `x = 2 sin(0.045 z)` and finishes at `z >= 90`. Reward is forward
Z progress, minus `0.002 * steering²` and `0.005 * lateral route error`, plus
20 for success or minus 10 for collision/offroad. Offroad includes leaving
safe terrain bounds or overturning. Vehicle-to-vehicle and obstacle collisions
terminate the involved active agents; ordinary wheel/terrain contacts do not.

Version 1 observations (`ml_cars/obs-v1`):

| Indices | Meaning |
| --- | --- |
| 0, 1 | Ego forward/lateral velocity / 10 m/s |
| 2 | Y angular velocity in rad/s |
| 3, 4 | Local X/Z position of an 8 m route lookahead target / 10 m |
| 5 | Remaining route Z distance / 100 m |
| 6, 7 | Previous throttle and steering |
| 8–12 | Five ray distances / sensing range, angles −0.8, −0.4, 0, 0.4, 0.8 rad |
| 13, 14 | Nearest visible forward vehicle's local X/Z / sensing range |
| 15, 16 | That vehicle's relative local X/Z velocity / 10 m/s |
| 17 | Nearby-vehicle presence mask |

Aware vehicles receive the nearest vehicle in the forward half-plane, within
`sensing_range`, with line-of-sight occlusion. Unaware vehicles have indices
13–17 zeroed and exclude all other vehicles from range queries. This changes
observations only: all bodies still collide. Policy observations omit terrain
material IDs and full world poses. `snapshot_json` is a separate, privileged
debug/critic interface; adapters do not put it into policy observations.

Scenario JSON rejects unknown keys and unsupported versions. See `scenarios/`
for native, learned, mixed-controller, car/truck, and aware/unaware examples.
`spawn` is `[x,z]`; seeded `spawn_jitter` offsets X. Optional vehicle
`parameters` override the car defaults (`mass`, `width`, `length`, `wheelbase`,
`steering`, `engine`); omit `parameters` to use the selected `kind`'s defaults.
Model paths resolve relative to the working directory.

## Deployment and reproducibility

`crates/sim` has no Python, Bevy, or controller dependency. Controllers implement
`Controller::action` (returning `Result<Action, String>`) and `reset`, with one instance per vehicle. The native
follower's integral state is independent per instance and reset with the world;
compatible neural weights share an `Arc` within a world. The supplied neural
policy is feedforward and has no hidden state. Each ONNX vehicle owns an
independent runtime session.

The portable JSON format stores schema version, exact observation/action schema
IDs, normalization `(observation - mean) / scale`, and row-major dense layers.
Every layer applies tanh, including the final two normalized actions. Shape,
finite-value, and schema validation happens on load. The JSON backend supports
only Linear/Tanh MLPs. ONNX policies use the separate backend described below;
recurrent neural policies remain future work.

Same seed + scenario + actions produce identical replay on the validated build
and platform. Tests compare exact state/observation values; cross-platform,
cross-compiler, and future-version bitwise reproducibility is not promised.
`Cargo.lock`, the report's build/dependency identifiers, and explicit scenario
versions document the tested inputs. The terrain generator is version 1.
Recordings are versioned **pose snapshots**, with scenario and seed; they are
for visualization, not action logs or physics checkpoints. Playback uses stored
poses and does not require the original model files. Live observations are not
reconstructed from recordings.

## ONNX policies

Basic **CPU, feedforward ONNX** support is included. `policy:<path>` selects ONNX
when the filename ends in `.onnx`; JSON models keep using the existing Rust MLP
backend. The CLI, Python `Batch` and adapters, `policy_actions`, and viewer all
use the same loader. The included `models/baseline.onnx` was converted from the
saved JSON weights without retraining. See its
[parity/evaluation report](models/baseline.onnx.report.json).

Install the export tools and CPU runtime, then locate the runtime library:

```sh
source .venv/bin/activate
uv pip install 'onnx>=1.18,<2' 'onnxscript>=0.3,<1' 'onnxruntime>=1.22,<2'
# PyTorch is also needed for export; install it as shown above.
export ORT_DYLIB_PATH="$(python -m ml_cars.onnx_runtime)"

# Run the included model entirely in Rust
cargo run --release -- run scenarios/onnx.json
cargo run --release -- run scenarios/onnx-mixed.json
cargo run -p ml-cars-viewer -- scenarios/onnx.json

# Convert a saved JSON policy and verify parity + eight complete evaluations
python -m ml_cars.export_onnx --input models/baseline.json --output models/baseline.onnx

# Or export both formats after training a new policy
python -m ml_cars.train --output models/new.json --onnx-output models/new.onnx

# Integration tests; ONNX tests skip when dependencies/runtime path are absent
python -m pytest -q
```

For a package installation, `uv pip install '.[onnx]'` installs the export/runtime
extra. The runtime-path helper is only a setup convenience; Rust never launches
Python to infer actions. For native deployment, ship/install the CPU ONNX Runtime
shared library and set `ORT_DYLIB_PATH` to its absolute path. A library named
`libonnxruntime.so`, `libonnxruntime.dylib`, or `onnxruntime.dll` next to the
executable or on the OS loader path also works. The `ort` crate is pinned to
`2.0.0-rc.13` with API 22 and runtime dynamic loading; CPU ONNX Runtime **1.22+**
is required (1.30.0 validated). No native runtime is downloaded during Cargo
builds. JSON/native-only scenarios and pose playback need no ONNX library.
The chosen library is process-global; set its path before first ONNX use.

The exporter uses `torch.onnx.export(..., dynamo=True)` with opset 18, embeds
weights and `(observations - mean) / scale` in one `.onnx` file, and names its
float32 tensors `observations[1,18]` and `actions[1,2]`. Rust also accepts a dynamic
leading batch dimension, but executes **one vehicle at a time**; the numerical
Python API loops over rows. Extra inputs/outputs, hidden-state tensors, other
dtypes, or different feature widths are rejected. Outputs must be finite and
within `[-1,1]`; they are never silently clipped.

Required ONNX custom metadata:

| Key | Value |
| --- | --- |
| `ml_cars.version` | `1` |
| `ml_cars.observation_schema` | `ml_cars/obs-v1` |
| `ml_cars.action_schema` | `throttle-steering/tanh-v1` |
| `ml_cars.normalization` | `embedded/v1` |
| `ml_cars.recurrent` | `false` |

Other feedforward ONNX graphs can use this contract if the CPU runtime supports
their operators. The supported export utility currently converts the project's
Linear/Tanh policies. GPU execution providers, recurrent policies, and grouping
multiple vehicles into a single ONNX inference call remain outside basic support.

## Validation and performance

```sh
cargo test
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
python -m pytest -q
RAYON_NUM_THREADS=4 cargo run --release -- bench 16 1000
```

Tests cover exact seeded replay, terrain seams/triangle interpolation, wheel
material lookup, traction response, unaware collisions, awareness filtering,
termination/truncation, independent controllers, batch equivalence, atomic
validation, and the Gymnasium/PettingZoo contracts. See [VALIDATION.md](VALIDATION.md)
for measured results and limitations. The physics is a research starting point:
raycast wheels, rigid ground, fixed suspension tuning, and simplified tire forces
have not been calibrated against real vehicles. Cameras, deformable terrain,
streaming maps, distributed training, and GPU/recurrent neural execution are
outside this implementation.
