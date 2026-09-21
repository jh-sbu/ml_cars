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
ONNX graph execution and recurrent neural models are not implemented.

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
cameras, distributed execution, and ONNX remain outside this implementation.
