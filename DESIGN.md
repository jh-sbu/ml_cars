# Vehicle training gym: requirements and recommended design

This document records the project requirements and the proposed implementation
approach discussed on September 18, 2026. Recommendations are starting decisions
to validate during implementation; they are not additional user requirements.

## Project requirements

The project is a training gym for simulated self-driving vehicles.

| ID | Requirement |
| --- | --- |
| R1 | Implement the simulation gym in Rust for performance. |
| R2 | Provide Python bindings suitable for developing and training controllers with PyTorch. |
| R3 | Support bringing controllers developed against the Python-accessible simulation into Rust execution where practical. |
| R4 | Support headless training. |
| R5 | Provide a GUI for visualizing trained models and their behavior. |
| R6 | Support different terrains over which vehicles drive. |
| R7 | Support multiple vehicles within a scenario. |
| R8 | Allow some vehicles to be aware of other vehicles, with awareness configurable independently of physical presence. |
| R9 | Support different vehicle types and distinct controllers for individual vehicles. |

## Working assumptions and open decisions

- Start with 3D terrain and control research using state vectors and simulated
  range sensors. Photorealistic camera training is not part of the initial scope.
- Begin with rigid terrain and configurable traction. Deformable soil, sinkage,
  and persistent ruts require additional physical models.
- Prioritize CPU simulation throughput across many independent worlds, with
  PyTorch optionally using a GPU for training and inference.
- The simulation must be Rust. The recommended deployed neural-inference path
  includes ONNX Runtime, a native dependency that is not itself pure Rust.
- Vehicle fidelity, target operating systems, expected vehicle counts, map sizes,
  and throughput targets remain to be established. Benchmark before committing
  to performance claims or more elaborate infrastructure.

## Recommended technology stack

| Layer | Recommendation | Reason |
| --- | --- | --- |
| Project organization | Cargo workspace | Separate the simulation, controllers, bindings, and viewer. |
| Simulation core | Standalone Rust library | Explicit stepping without a window, Python interpreter, or render loop. |
| Physics | `rapier3d` | Rigid bodies, collisions, joints, ray queries, heightfields, and meshes. |
| Vehicle dynamics | Project-owned Rust module over Rapier | Control steering, drivetrain, suspension, tire forces, and actuator constraints. |
| Parallel worlds | `rayon` | Parallelize independent scenarios across CPU cores. |
| Python interface | PyO3, `numpy`, and maturin | Package the Rust extension and exchange numerical arrays with Python. |
| Environment adapters | Gymnasium and PettingZoo Parallel API | Single-agent compatibility and simultaneous multi-agent stepping. |
| Learning | PyTorch | Policy experimentation, optimization, and checkpoints. |
| Rust neural inference | `ort` with ONNX Runtime | Execute exported policies without a Python runtime. |
| Viewer | Bevy with `bevy_egui` | 3D visualization, controls, debugging overlays, and replay. |
| Configuration | Serde with TOML or JSON | Versioned scenarios, vehicle parameters, and experiment configuration. |

Pin compatible dependency versions when implementing the workspace, particularly
Bevy and its integrations. No versions or dependencies have been selected in code
as part of this document.

## Architecture

The simulation owns authoritative state and simulation time. Training, native
evaluation, and visualization all use the same core behavior.

```text
TRAINING
Python / PyTorch
       |
       | batched observations and actions
       v
PyO3 bindings
       |
       v
Rust simulation core <----- Scenario definitions
       |                    Terrain assets
       |                    Vehicle definitions
       |
       +---- snapshots ----> Bevy GUI / replay


RUST EVALUATION
Exported ONNX policies or native Rust controllers
       |
       | observations and actions
       v
Same Rust simulation core
       |
       +---- snapshots ----> Bevy GUI / replay
```

Recommended initial layout:

```text
crates/
  sim/           # Worlds, physics, terrain, vehicles, sensors, and task rules
  controllers/   # Controller interfaces, native controllers, optional ONNX
  python/        # PyO3 bindings and batched environment API
  viewer/        # Bevy application
python/
  ml_cars/       # Gymnasium/PettingZoo adapters and training code
scenarios/       # Scenario and experiment definitions
assets/          # Terrain and visualization assets
```

The simulation crate must not depend on the viewer or Python bindings. Start with
ordinary Rust data structures and explicit systems; do not require a Bevy ECS
world inside every training environment. Split further crates only when the code
or dependency boundaries justify it.

### Worlds and stepping

Each world owns its physics state, vehicles, sensors, task state, and seeded random
state. A batch is a collection of independent worlds.

Expose explicit `reset` and `step` operations. An observation from time `t` is used
to select actions for time `t`; all actions are collected before any are applied.
Advance physics, then return observations and rewards for the resulting state.

1. Obtain each active agent's observation of the current state.
2. Collect external and native-controller actions for that same instant.
3. Apply the actions together.
4. Advance a fixed number of physics substeps.
5. Compute observations, rewards, per-agent termination, and scenario completion.

Physics frequency and policy frequency are separate. An initial experiment could
use 120 Hz physics and 20 Hz control, with six physics substeps per action. These
are proposed defaults to validate, not established fidelity requirements.

Specify what happens when an agent finishes: removal, stopped vehicle, or continued
scripted motion. Keep physical presence distinct from participation in learning.
Define missing-action handling explicitly rather than silently applying stale
actions forever.

### Vehicles and controllers

Separate three responsibilities:

- **Vehicle definition:** mass, dimensions, wheel layout, actuator limits,
  drivetrain, suspension parameters, and sensor configuration.
- **Vehicle dynamics:** turn actuator commands into forces and physical motion.
- **Controller:** turn observations and controller state into commands.

A scenario assigns a controller and observation configuration to each vehicle.
Controllers may be external Python policies, native Rust algorithms, or exported
neural policies. Multiple vehicles may share model weights while retaining
independent controller state, including recurrent hidden state.

Rapier's raycast vehicle controller is a reasonable initial dynamics model. Keep
it behind a project-owned interface and validate its behavior against the needs
of the gym. It is not an automatic guarantee of automotive research fidelity.
Account for the limitations of raycast wheels around curbs, narrow obstacles,
and abrupt surface changes before expanding the terrain difficulty.

### Observations and agent awareness

The world always contains the physically simulated vehicles. Observation builders
determine which information each controller receives.

Possible observation inputs include ego motion, actuator state, route information,
range measurements, and nearby vehicles. Configure sensing range, field of view,
occlusion, noise, and optional communication independently per vehicle.

Privileged world state must be a separate interface from policy observations. This
allows centralized training critics without exposing hidden information to the
deployed controller. Ground-truth terrain materials and semantic maps are likewise
available to task logic but should enter observations only when explicitly allowed.

### Python and batch execution

Provide a convenient environment API and a numerical batch API over the same core.
Use Gymnasium for a single learning vehicle with background traffic, and
PettingZoo's Parallel API for simultaneous multi-agent actions. Different agents
may have different observation and action spaces.

For throughput:

- Exchange contiguous numerical arrays once per batch step rather than calling
  Python separately for every vehicle or physics substep.
- Release the GIL while Rust advances the batch; Rust worker threads should not
  need Python callbacks during simulation.
- Group policy inference by compatible model and tensor schema. Use stable agent
  IDs, active-agent masks, and explicit indexing for variable populations.
- Parallelize across worlds first. Control thread counts to avoid oversubscribing
  the CPU with nested physics, inference, and environment thread pools.
- Define ownership and buffer lifetimes before pursuing zero-copy optimizations.
  NumPy interoperability does not eliminate CPU-to-GPU transfers.

Distinguish termination from time-limit truncation. Preserve final observations
when resetting finished environments so learning algorithms bootstrap correctly.

### Moving controllers into Rust

Conventional controllers can be implemented natively behind the controller
interface. Neural controllers can be trained in PyTorch, exported using its
`torch.export`-based ONNX exporter, and executed through `ort`.

Package the model with observation order, normalization parameters, action scaling,
tensor shapes, schema version, and recurrent-state rules. Validate operator support
and compare Python and Rust outputs on identical observations before deployment.
Reset controller state at episode boundaries.

Moving inference into Rust removes Python orchestration overhead but does not
guarantee faster tensor operations. Benchmark end-to-end execution, including
batching and data movement. Do not make TorchScript the default export path for
a new project.

### GUI and headless execution

The Bevy viewer consumes state snapshots and sends explicit pause, step, reset,
and playback commands. Rendering frequency does not determine physics frequency.
Do not create a second authoritative physics simulation in the viewer.

Start with in-process visualization of the core and recorded playback. Remote
viewing or a separate viewer process can follow if needed; a network service is
not necessary for the MVP.

Headless state-based training initializes no graphics. Camera-based observations,
if added later, need an explicit offscreen rendering path even in headless mode.

## Terrain representation

Use tiled heightfields for most ground, additional static geometry where needed,
and independent physical-surface and semantic layers.

| Layer | Contents | Consumers |
| --- | --- | --- |
| Geometry | Elevation, slopes, road surfaces, rocks, bridges | Collision detection, wheel queries, range sensors |
| Physical surface | Material, grip parameters, rolling resistance, wetness | Vehicle dynamics |
| Semantics | Lanes, boundaries, drivable regions, surface labels | Task rules, rewards, explicitly permitted observations |
| Appearance | Textures, colors, vegetation, visual detail | Viewer and camera sensors |

Layers use a common coordinate convention and stable surface identifiers but may
have different resolutions. Adopt meters and a documented axis convention; a
Y-up world with terrain in the X-Z plane is a practical choice.

### Heightfields

A heightfield stores one elevation per sample on a regular horizontal grid:

```text
height[x_index, z_index] -> elevation in meters

HeightfieldTile
  origin
  sample_spacing
  sample_dimensions
  heights
  material_map
  semantic_map
```

Each cell consists of two collision triangles. Store enough information to map
samples unambiguously into world space, including the triangle split convention.

A 64 by 64 meter tile sampled every 0.5 meters has 129 by 129 height samples,
including both boundaries. Its raw `f32` elevation data occupies 66,564 bytes,
approximately 65 KiB, before collision acceleration data, maps, or render meshes.
This is an example layout, not a required tile size.

Heightfields are suitable for hills, broad ditches, banks, and smooth road profiles.
They cannot represent multiple surfaces at the same horizontal position, such as
a bridge and the road beneath it. Narrow curbs and small potholes need finer local
geometry or additional shapes.

Choose physical resolution based on the smallest feature that needs to affect
vehicle motion. Visual texture resolution is independent. Suspension and clearance
queries must agree with the collision triangles: bilinear interpolation can
produce a different surface from the piecewise triangular collision geometry.

### Additional geometry and roads

Use static triangle meshes for bridges, overhangs, and complex road surfaces.
Use simple solid shapes or suitable meshes for curbs and barriers. Movable debris
belongs in the world as dynamic objects rather than immutable terrain.

Road authoring can eventually use splines with width, elevation, banking, and
material attributes. An asset-building step generates collision geometry, render
geometry, and lane metadata from that source. A spline editor is not needed for
the first milestone.

Where a road mesh replaces the ground, remove or lower the underlying ground to
avoid coincident collision surfaces. Under bridges, retain the lower ground.
Rendering and collision geometry should derive from the same source asset.

### Materials and contact queries

Store material IDs in a grid for heightfields and per triangle or in a surface map
for mesh geometry. Keep sharp material boundaries where appropriate; interpolate
continuous fields such as wetness separately from categorical material IDs.

A proposed surface-contact result is:

```text
SurfaceContact
  position
  normal
  surface_id
  material_id
  physical_parameters
```

Retain the actual hit surface, not just horizontal coordinates. The bridge deck
and dirt below it may have different materials at the same X-Z location.

Material parameters feed the chosen tire model. Initial inputs can include grip
scaling and rolling resistance; more sophisticated slip-dependent behavior can
follow. A material label such as gravel has no physical effect until the dynamics
implementation consumes its parameters.

The integration depends on the wheel model:

- Physical wheel colliders use solver contacts. Rapier contact modification can
  supply spatially varying friction where needed.
- Raycast or shape-cast wheels obtain a surface hit and explicitly incorporate
  its properties into suspension and tire-force calculations. Do not assume
  setting ground-collider friction alone implements the desired tire behavior.

### Shared assets and per-world state

```text
TerrainAsset                     TerrainInstance
  height tiles                     shared asset reference
  static mesh geometry             episode-specific overrides
  material maps                    wetness / ice patches
  material definitions             deformation state, if supported
  semantic maps                    this world's collider mappings
```

Share immutable source data across environments, for example through `Arc`.
Each world retains its own physics bookkeeping and mutable terrain state. Share
immutable collision shapes where the selected APIs permit it; do not assume all
per-world physics structures can be shared.

Initially load bounded scenarios completely. Add streaming only if map sizes
require it. Streaming must cover every vehicle's collision and sensor needs,
independently of camera position, and must not introduce timing-dependent physics.

Neighboring tiles need identical boundary elevations and consistent seams.
Keep collision detail fixed during an episode even if visual detail changes with
camera distance. Test driving and sensor queries across tile boundaries.

### Generation and reproducibility

Support authored and seeded procedural terrains through the same runtime asset
format. Useful generation parameters include slope, roughness, road curvature,
banking, obstacle density, and material distribution.

Prefer generation that creates meaningful courses: lay out routes, shape terrain,
assign materials, and validate spawns and required paths. Store generator version,
parameters, and seed; retain generated assets when exact long-term reproduction
is important. A seed alone does not reproduce a map after its generator changes.

### Deformable ground

The initial terrain is rigid. Ice and simplified gravel can affect motion through
tire parameters on fixed geometry. Mud, sand sinkage, rutting, and persistent
deformation need additional state and force models. Updating heights or reducing
friction alone does not establish a validated soil simulation.

## Recommended MVP

Build a small end-to-end system before increasing simulation fidelity or scale.
The first integration milestone proves the training-to-Rust workflow; the full
MVP then exercises the multi-vehicle requirements.

### Milestone 1: complete one training and deployment loop

1. Establish the workspace and explicit, fixed-step simulation API.
2. Implement one configurable wheeled vehicle using a validated initial raycast
   dynamics model and a native path-following baseline controller.
3. Provide two bounded terrain configurations with different geometry and traction,
   plus a mixed-material area for testing wheel-contact material lookup.
4. Implement a route-following task with observations, rewards, collision handling,
   episode termination, time limits, and seeded resets.
5. Expose batched Python stepping and a Gymnasium adapter.
6. Train one small PyTorch policy; export it and run it through the Rust controller
   interface with numerical parity checks.
7. Add a Bevy viewer for live evaluation and basic recorded playback, with pause,
   single-step, reset, and useful vehicle/sensor overlays.

### Milestone 2: exercise all core scenario requirements

1. Add a second vehicle configuration with meaningfully different dynamics, such
   as mass, wheelbase, and steering limits. A second mechanics implementation is
   unnecessary unless configuration alone cannot express the needed differences.
2. Run multiple vehicles in one world with independent controller assignments and
   controller state, including a mixture of native and learned controllers.
3. Add configurable nearby-vehicle observations and demonstrate both aware and
   unaware controllers while retaining physical collisions for all vehicles.
4. Provide a PettingZoo Parallel adapter and verify simultaneous stepping and
   correct agent lifecycle behavior.
5. Measure batch throughput and memory use with a representative mix of terrains,
   vehicle counts, and controllers. Use the measurements to set scaling targets.

### MVP acceptance checks

- The same scenario and core stepping logic run headlessly and in the viewer.
- Resetting with the same seed and replaying the same actions reproduces behavior
  under the documented build and platform constraints.
- Vehicle behavior responds to intended geometry and traction changes; contact
  material lookup and tile seams have targeted checks.
- Multiple vehicles can use distinct controllers and independent recurrent state.
- Awareness configuration changes observations without removing physical vehicles
  or leaking privileged state.
- Batch results agree with equivalent individually stepped worlds, within the
  defined numerical expectations.
- Python environment adapters pass their applicable API checks and preserve final
  observations, termination, and truncation semantics.
- Exported policy outputs agree with PyTorch within a specified tolerance on a
  representative observation set; Rust evaluation completes without Python.
- The viewer can inspect a trained policy's behavior and replay a recorded run.
- Throughput and memory measurements are recorded; no speedup is presumed merely
  because a component is implemented in Rust.

### Defer beyond the MVP

- Photorealistic cameras and large-scale GPU rendering.
- Deformable soil, detailed tire contact patches, and high-fidelity powertrains.
- Terrain streaming, very large maps, and an interactive road/terrain editor.
- Distributed training orchestration and remote viewer services.
- Fully general plugin systems or dynamic loading of arbitrary Rust controllers.
- More complex cooperative training algorithms beyond what is needed to prove
  the environment and controller interfaces.

## Reproducibility and validation

Record scenario and asset versions, seeds, actions, model metadata, simulation
configuration, dependency/build identifiers, and checkpoints as appropriate.
Define whether a replay is a state recording for visualization or action replay
through physics; those serve different reproducibility needs.

Rapier's determinism depends on initialization order, configuration, and build or
platform conditions. Deterministic physics alone does not make sensors, random
generation, controller inference, or asynchronous infrastructure deterministic.
Test the complete stepping path and state the supported reproducibility scope.

## Reference documentation

- Rapier overview: <https://rapier.rs/>
- Rapier colliders, heightfields, meshes, and friction: <https://rapier.rs/docs/user_guides/rust/colliders/>
- Rapier raycast vehicle controller: <https://docs.rs/rapier3d/latest/rapier3d/control/struct.DynamicRayCastVehicleController.html>
- Rapier physics hooks: <https://docs.rs/rapier3d/latest/rapier3d/pipeline/trait.PhysicsHooks.html>
- Rapier determinism: <https://rapier.rs/docs/user_guides/templates/determinism/>
- Rayon: <https://docs.rs/rayon/latest/rayon/>
- PyO3: <https://docs.rs/pyo3/latest/pyo3/>
- Rust NumPy bindings: <https://pyo3.github.io/rust-numpy/>
- maturin: <https://www.maturin.rs/>
- Gymnasium environment API: <https://gymnasium.farama.org/api/env/>
- PettingZoo Parallel API: <https://pettingzoo.farama.org/main/api/parallel/>
- PyTorch ONNX export: <https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html>
- ort: <https://github.com/pykeio/ort>
- Bevy plugins and headless configuration: <https://bevy.org/learn/quick-start/getting-started/plugins/>
- bevy_egui: <https://docs.rs/bevy_egui/latest/bevy_egui/>
