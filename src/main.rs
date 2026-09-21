use ml_cars_controllers::{Batch, Recording, Runner, load_policy};
use ml_cars_sim::{OBS_DIM, Scenario};
use std::{env, fs, time::Instant};
fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str).unwrap_or("help") {
        "run" => {
            let path = args
                .get(2)
                .map(String::as_str)
                .unwrap_or("scenarios/flat.json");
            let scenario =
                Scenario::from_json(&fs::read_to_string(path).map_err(|e| e.to_string())?)?;
            let seed = args
                .get(3)
                .map(|x| x.parse::<u64>())
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or(42);
            let mut runner = Runner::new(scenario.clone(), seed)?;
            let mut recording = Recording {
                version: 1,
                scenario,
                seed,
                frames: vec![runner.world.snapshot()],
            };
            let mut total = vec![0.; runner.world.scenario.vehicles.len()];
            while runner.world.active().iter().any(|x| *x) {
                let r = runner.step(&vec![None; total.len()])?;
                for (t, r) in total.iter_mut().zip(&r.rewards) {
                    *t += r;
                }
                recording.frames.push(runner.world.snapshot());
            }
            println!(
                "{}",
                serde_json::json!({"steps":runner.world.steps,"rewards":total,"reasons":runner.world.initial().reasons,"final_state":runner.world.snapshot()})
            );
            if let Some(path) = args.get(4) {
                recording.save(path)?;
            }
        }
        "bench" => {
            let count = args
                .get(2)
                .map(|x| x.parse::<usize>())
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or(16);
            let steps = args
                .get(3)
                .map(|x| x.parse::<u32>())
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or(200);
            if count == 0 || steps == 0 {
                return Err("batch size and steps must be positive".into());
            }
            let scenario = Scenario::from_json(
                &fs::read_to_string(
                    args.get(4)
                        .map(String::as_str)
                        .unwrap_or("scenarios/mixed.json"),
                )
                .map_err(|e| e.to_string())?,
            )?;
            let vehicle_count = scenario.vehicles.len();
            let mut worlds = Vec::new();
            for i in 0..count {
                let mut cfg = scenario.clone();
                cfg.terrain = if i % 2 == 0 {
                    ml_cars_sim::TerrainKind::Flat
                } else {
                    ml_cars_sim::TerrainKind::Hills
                };
                worlds.push(Runner::new(cfg, i as u64)?);
            }
            let mut batch = Batch { worlds };
            let actions = vec![vec![None; vehicle_count]; count];
            let start = Instant::now();
            for k in 0..steps {
                batch.step(&actions)?;
                for (i, w) in batch.worlds.iter_mut().enumerate() {
                    if !w.world.active().iter().any(|x| *x) {
                        w.reset(k as u64 + 100 + i as u64)?;
                    }
                }
            }
            let elapsed = start.elapsed().as_secs_f64();
            let memory = fs::read_to_string("/proc/self/status")
                .unwrap_or_default()
                .lines()
                .find(|x| x.starts_with("VmHWM:"))
                .unwrap_or("unavailable")
                .to_string();
            println!(
                "{}",
                serde_json::json!({"worlds":count,"vehicles_per_world":vehicle_count,"terrains":"alternating flat/hills","controllers":scenario.vehicles.iter().map(|v| &v.controller).collect::<Vec<_>>(),"steps_per_world":steps,"seconds":elapsed,"world_steps_per_second":count as f64*steps as f64/elapsed,"peak_memory":memory,"profile":if cfg!(debug_assertions){"dev"}else{"release"}})
            );
        }
        "infer" => {
            let mut policy = load_policy(args.get(2).ok_or("infer needs model path")?)?;
            let observations: Vec<[f32; OBS_DIM]> = serde_json::from_str(
                &fs::read_to_string(args.get(3).ok_or("infer needs observations JSON path")?)
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            println!(
                "{}",
                serde_json::to_string(
                    &observations
                        .iter()
                        .map(|o| policy.action(o))
                        .collect::<Result<Vec<_>, _>>()?
                )
                .map_err(|e| e.to_string())?
            );
        }
        _ => println!(
            "ml_cars run [scenario.json] [seed] [recording.json]\nml_cars bench [worlds=16] [steps=200] [scenario=scenarios/mixed.json]\nml_cars infer <policy.json|policy.onnx> <observations.json>\nViewer: cargo run -p ml-cars-viewer -- [scenario.json | --replay recording.json]"
        ),
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
