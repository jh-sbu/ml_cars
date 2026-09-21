use ml_cars_controllers::*;
use ml_cars_sim::*;
#[test]
fn batch_matches_serial_worlds() {
    let cfg = Scenario::default();
    let mut batch = Batch::new(cfg.clone(), &[4, 5, 6]).unwrap();
    let mut serial: Vec<_> = (4..7)
        .map(|s| Runner::new(cfg.clone(), s).unwrap())
        .collect();
    for _ in 0..50 {
        let results = batch.step(&vec![vec![None]; 3]).unwrap();
        for (i, w) in serial.iter_mut().enumerate() {
            let result = w.step(&[None]).unwrap();
            assert_eq!(results[i].observations, result.observations);
            assert_eq!(results[i].rewards, result.rewards);
        }
    }
}
#[test]
fn native_completes_both_terrains_and_reset_clears_controller() {
    for terrain in [TerrainKind::Flat, TerrainKind::Hills] {
        let mut r = Runner::new(
            Scenario {
                terrain,
                ..Default::default()
            },
            42,
        )
        .unwrap();
        while r.world.active()[0] {
            r.step(&[None]).unwrap();
        }
        assert_eq!(r.world.initial().reasons[0], "success");
        let end = r.world.observations();
        r.reset(42).unwrap();
        while r.world.active()[0] {
            r.step(&[None]).unwrap();
        }
        assert_eq!(end, r.world.observations());
    }
}
#[test]
fn invalid_batch_does_not_advance_any_world() {
    let mut cfg = Scenario::default();
    cfg.vehicles[0].controller = "external".into();
    let mut b = Batch::new(cfg, &[1, 2]).unwrap();
    assert!(
        b.step(&[vec![Some(Action::default())], vec![None]])
            .is_err()
    );
    assert!(b.worlds.iter().all(|w| w.world.steps == 0));
}
#[test]
fn controller_state_is_independent() {
    let mut a = PathFollower::default();
    let mut b = PathFollower::default();
    let mut obs = [0.; OBS_DIM];
    obs[0] = 0.48;
    obs[4] = 0.8;
    let first = b.action(&obs).unwrap();
    for _ in 0..20 {
        a.action(&obs).unwrap();
    }
    assert!(a.action(&obs).unwrap().throttle > first.throttle);
    b.reset();
    assert_eq!(b.action(&obs).unwrap().throttle, first.throttle);
}
#[test]
fn malformed_policy_shapes_are_rejected() {
    let p = Policy {
        version: 1,
        observation_schema: "ml_cars/obs-v1".into(),
        action_schema: "throttle-steering/tanh-v1".into(),
        mean: vec![0.; OBS_DIM],
        scale: vec![1.; OBS_DIM],
        layers: vec![Layer {
            weights: vec![vec![0.; OBS_DIM]; 2],
            bias: vec![0.; 2],
        }],
    };
    assert!(p.validate().is_ok());
    let mut invalid = p;
    invalid.layers[0].weights[0].pop();
    assert!(invalid.validate().is_err());
}

#[test]
fn recording_round_trip_and_schema_validation() {
    let scenario = Scenario::default();
    let runner = Runner::new(scenario.clone(), 7).unwrap();
    let recording = Recording {
        version: 1,
        scenario,
        seed: 7,
        frames: vec![runner.world.snapshot()],
    };
    let path = std::env::temp_dir().join(format!("ml-cars-recording-{}.json", std::process::id()));
    let path_str = path.to_str().unwrap();
    recording.save(path_str).unwrap();
    let loaded = Recording::load(path_str).unwrap();
    assert_eq!(
        serde_json::to_string(&recording.frames).unwrap(),
        serde_json::to_string(&loaded.frames).unwrap()
    );
    let mut invalid = recording;
    invalid.frames.clear();
    invalid.save(path_str).unwrap();
    assert!(Recording::load(path_str).is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn learned_and_native_vehicles_run_together() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut scenario =
        Scenario::from_json(&std::fs::read_to_string(root.join("scenarios/mixed.json")).unwrap())
            .unwrap();
    scenario.vehicles[0].controller =
        format!("policy:{}", root.join("models/baseline.json").display());
    let mut runner = Runner::new(scenario, 42).unwrap();
    for _ in 0..100 {
        runner.step(&[None, None, None]).unwrap();
    }
    assert_eq!(runner.world.snapshot().vehicles.len(), 3);
    assert!(runner.world.snapshot().vehicles[0].position[2] > 10.);
    assert!(runner.world.snapshot().vehicles[1].position[2] > 25.);
}
