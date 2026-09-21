# MVP validation

Validated on September 21, 2026 on Linux x86_64 (kernel 7.2.3), AMD Ryzen 7
5825U, Rust 1.97.1, Python 3.12.13. Cargo versions are locked; Python versions
are recorded in `python/requirements-validated.txt`.

## Acceptance evidence

| Check | Result |
| --- | --- |
| Headless Rust core and viewer share stepping | Viewer uses the same `Runner`/`World`; live evaluation launched successfully |
| Seeded reset and identical action replay | Exact snapshot equality on this build/platform |
| Geometry, traction, material contacts, tile seams | Targeted Rust tests pass, including driving across a seam and mixed materials under different wheels |
| Car/truck dynamics and custom parameters | Both types complete the hilly route; doubling chassis mass changes acceleration in a targeted test |
| Independent controllers and awareness | Stateful native controllers reset independently; learned/native mixture runs; unaware vehicles still collide and cannot sense other vehicles |
| Batch vs serial execution | Exact observation/reward equality; invalid actions do not partially advance a batch |
| Gymnasium/PettingZoo API and lifecycle | API checkers pass; tests cover final observations, truncation, early agent removal, and retained physical bodies |
| Trained policy executes in Rust | Included policy completes 8/8 held-out flat/hilly evaluations; standalone inference and full native evaluation run without Python |
| Numerical parity | Maximum absolute error `2.1606683731079102e-7` on 512 demonstration observations, tolerance `2e-5`; an additional normalized random-input test passes |
| Viewer controls and playback | Two application-system tests cover pause, single step, seeded reset, playback and rewind without advancing replay physics |
| Actual rendering | Live learned-policy evaluation and recorded playback both exit successfully; captured screenshots inspected for terrain, vehicle, route, HUD, and live sensor rays |
| Throughput and memory | Measurements below; no general performance claim |

Commands run:

```sh
cargo test                         # 16 simulation/controller integration tests
cargo test -p ml-cars-viewer        # 2 viewer application-system tests
cargo clippy --workspace --all-targets -- -D warnings
python -m pytest -q                # 7 tests including both environment API checkers
python -m ml_cars.train --output models/baseline.json
cargo run --release -- run scenarios/learned.json 42 /tmp/ml-cars-learned-replay.json
```

Gymnasium emits two advisory warnings for deliberately unbounded observation
spaces (linear and angular velocities). No check fails. The exported policy is
feedforward behavior cloning, with a schema-checked JSON deployment format;
Basic feedforward CPU ONNX execution has since been added; see the ONNX
validation section below. Recurrent neural models are not implemented.

## Throughput

Release builds, one measured run per configuration, 2,000 policy steps per world,
120 Hz physics / 20 Hz policy frequency. Each world contains one learned car,
one aware native truck, and one unaware native car, using `scenarios/mixed.json`.
Worlds alternate flat/hilly terrain; the single-world case is flat. Episodes
reset when all agents finish. Initialization is excluded from elapsed time;
resets during the run are included. Finished physical vehicles remain present.

| Worlds | Rayon threads | World-steps/s | Elapsed | Peak process RSS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 10,612 | 0.188 s | 4,976 KiB |
| 16 | 4 | 31,904 | 1.003 s | 7,864 KiB |

Reproduce using:

```sh
RAYON_NUM_THREADS=1 cargo run --release -- bench 1 2000
RAYON_NUM_THREADS=4 cargo run --release -- bench 16 2000
```

These are point measurements on this laptop, not statistical benchmarks or
speedup claims. They include native learned-policy inference and Rust world
stepping, but exclude Python tensor transfer, PyTorch training, and rendering.
RSS comes from Linux `/proc/self/status` (`VmHWM`). A conservative initial
regression target on this host is 10,000 combined world-steps/s and under 32 MiB
peak RSS for the 16-world/four-thread workload; repeat measurements before
using these figures to choose deployment hardware or increase scale.

## Known boundaries

The physics tests establish coherent behavior within the MVP, not automotive
fidelity. Suspension/tire parameters are uncalibrated; rigid ground and raycast
wheels have the limitations described in DESIGN.md. Neural training covers
route following on two terrains, not general traffic safety. The traffic demo
can intentionally end in collisions or timeouts, including when a finished
vehicle blocks another vehicle. Braked finished vehicles may slide on slopes.

Determinism is checked within this build/platform. Pose recordings replay stored
state; they are not action logs or checkpoints. Model files are unnecessary for
playback. Recurrent-controller interfaces can own per-instance state, but the
supplied learned policy has no recurrence. Terrain streaming, deformable soil,
cameras, distributed execution, and recurrent/GPU ONNX execution remain outside this implementation.

## ONNX validation

Basic CPU ONNX support was validated on the same host with `ort 2.0.0-rc.13`
(API 22), ONNX Runtime 1.30.0, ONNX 1.23.0, onnxscript 0.7.2, and PyTorch
2.14.0+cpu. The original saved JSON weights were converted without retraining.
The single ONNX file contains the normalization, weights, and policy metadata.
Its [report](models/baseline.onnx.report.json) records dependency/model hashes,
512-sample parity, and eight successful held-out course evaluations.

- Maximum PyTorch/Rust ONNX absolute error: **1.6391277313232422e-7**
  (tolerance `2e-5`). Verification includes a standalone native process.
- Native ONNX evaluation on the hilly course succeeds in 388 steps with seed 42.
- **19 Rust tests** pass (17 core/controller tests plus two viewer tests).
- **26 Python tests** pass with ONNX dependencies and `ORT_DYLIB_PATH` configured.
  Checks include PyTorch/JSON/ONNX parity, re-exporting, tensor names/shapes/dtypes,
  metadata, rejected recurrent inputs, invalid outputs, missing runtimes, mixed
  JSON/native/ONNX controllers, batch/serial equivalence, and resets.
- Controller errors and invalid inferred actions do not partially advance batch
  physics. Reset after a runtime failure before retrying: preparing actions may
  have updated native controller state.
- A one-epoch training smoke test exercises `--onnx-output` and both parity paths;
  its untrained driving scores are not included in the trained-policy results.
- Workspace Clippy passes with warnings denied. Live ONNX rendering was launched
  and the screenshot inspected for the vehicle, route, HUD and sensor rays.
  Recorded ONNX playback was also launched successfully with an intentionally
  invalid runtime-library path, confirming that pose playback needs no runtime.

A release comparison used 16 worlds, four Rayon workers, 2,000 steps/world,
alternating flat/hilly terrain, and three vehicles/world. Both runs used the
same scenario and seeds: `scenarios/onnx-mixed.json`, with only the first
vehicle's model switched between `.json` and `.onnx`. The other vehicles were
one native truck and one JSON-policy car. Each ONNX vehicle has its own CPU
session configured with one intra-op and one inter-op thread.

| First vehicle's backend | World-steps/s | Elapsed | Peak process RSS |
| --- | ---: | ---: | ---: |
| Rust JSON MLP | 33,687 | 0.950 s | 8,400 KiB |
| ONNX Runtime CPU | 30,800 | 1.039 s | 37,740 KiB |

These are single-run point measurements, with initialization excluded and
in-run resets included. ONNX adds runtime/session memory and did not improve
throughput for this small MLP. This comparison uses a different controller mix
from the original MVP benchmark above. GPU execution, recurrent inputs, and
cross-vehicle inference batching remain outside basic ONNX support.
