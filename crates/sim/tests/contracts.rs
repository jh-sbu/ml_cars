use ml_cars_sim::*;
fn drive(w: &mut World, n: usize, a: Action) {
    for _ in 0..n {
        if w.active().iter().any(|x| *x) {
            w.step(&vec![a; w.scenario.vehicles.len()]).unwrap();
        }
    }
}
#[test]
fn seeded_reset_and_action_replay_are_exact() {
    let mut w = World::new(Scenario::default(), 91).unwrap();
    let initial = serde_json::to_string(&w.snapshot()).unwrap();
    drive(
        &mut w,
        50,
        Action {
            throttle: 0.6,
            steering: 0.1,
        },
    );
    let end = serde_json::to_string(&w.snapshot()).unwrap();
    w.reset(91).unwrap();
    assert_eq!(initial, serde_json::to_string(&w.snapshot()).unwrap());
    drive(
        &mut w,
        50,
        Action {
            throttle: 0.6,
            steering: 0.1,
        },
    );
    assert_eq!(end, serde_json::to_string(&w.snapshot()).unwrap());
    w.reset(92).unwrap();
    assert_ne!(initial, serde_json::to_string(&w.snapshot()).unwrap());
}
#[test]
fn traction_changes_motion_and_contacts_are_per_wheel() {
    let mut cfg = Scenario {
        spawn_jitter: 0.,
        ..Default::default()
    };
    cfg.vehicles[0].spawn = [4., 32.];
    let mut ice = World::new(cfg.clone(), 0).unwrap();
    cfg.mixed_material = false;
    let mut asphalt = World::new(cfg, 0).unwrap();
    let action = Action {
        throttle: 1.,
        steering: 0.,
    };
    drive(&mut ice, 35, action);
    drive(&mut asphalt, 35, action);
    assert!(asphalt.snapshot().vehicles[0].speed > ice.snapshot().vehicles[0].speed * 1.5);
    let mut cfg = Scenario {
        spawn_jitter: 0.,
        ..Default::default()
    };
    cfg.vehicles[0].spawn = [0., 36.];
    let w = World::new(cfg, 0).unwrap();
    let m = &w.snapshot().vehicles[0].materials;
    assert!(m.contains(&0) && m.contains(&2));
}
#[test]
fn terrain_seams_and_triangle_height_agree() {
    for kind in [TerrainKind::Flat, TerrainKind::Hills] {
        let t = Terrain { kind, mixed: true };
        let meshes = t.meshes();
        for z in 0..=32 {
            assert_eq!(meshes[0].vertices[z * 9 + 8], meshes[1].vertices[z * 9]);
        }
        for m in meshes {
            for tri in m.triangles {
                let a = m.vertices[tri[0] as usize];
                let b = m.vertices[tri[1] as usize];
                let c = m.vertices[tri[2] as usize];
                let p = [
                    (a[0] + b[0] + c[0]) / 3.,
                    (a[1] + b[1] + c[1]) / 3.,
                    (a[2] + b[2] + c[2]) / 3.,
                ];
                assert!((t.height(p[0], p[2]) - p[1]).abs() < 1e-5);
            }
        }
    }
}
#[test]
fn awareness_filters_all_vehicle_sensors_without_removing_bodies() {
    let mut cfg = Scenario {
        spawn_jitter: 0.,
        ..Default::default()
    };
    cfg.vehicles.push(VehicleConfig {
        id: "front".into(),
        spawn: [0., 8.],
        ..Default::default()
    });
    let aware = World::new(cfg.clone(), 0).unwrap();
    cfg.vehicles[0].aware = false;
    let blind = World::new(cfg, 0).unwrap();
    assert_eq!(aware.observations()[0][17], 1.);
    assert_eq!(blind.observations()[0][17], 0.);
    assert!(aware.observations()[0][10] < blind.observations()[0][10]);
    assert_eq!(
        serde_json::to_string(&aware.snapshot()).unwrap(),
        serde_json::to_string(&blind.snapshot()).unwrap()
    );
}
#[test]
fn unaware_vehicles_still_collide() {
    let mut cfg = Scenario {
        spawn_jitter: 0.,
        ..Default::default()
    };
    cfg.vehicles[0].aware = false;
    cfg.vehicles.push(VehicleConfig {
        id: "front".into(),
        spawn: [0., 6.],
        aware: false,
        ..Default::default()
    });
    let mut w = World::new(cfg, 0).unwrap();
    let mut hit = false;
    for _ in 0..160 {
        let r = w
            .step(&[
                Action {
                    throttle: 1.,
                    steering: 0.,
                },
                Action {
                    throttle: -1.,
                    steering: 0.,
                },
            ])
            .unwrap();
        if r.reasons.iter().any(|r| r == "collision") {
            hit = true;
            break;
        }
    }
    assert!(hit);
}
#[test]
fn truncation_is_distinct_and_finished_bodies_remain() {
    let mut w = World::new(
        Scenario {
            max_steps: 1,
            ..Default::default()
        },
        0,
    )
    .unwrap();
    let r = w.step(&[Action::default()]).unwrap();
    assert!(!r.terminated[0] && r.truncated[0] && !r.active[0]);
    assert_eq!(w.snapshot().vehicles.len(), 1);
    assert!(w.step(&[Action::default()]).is_err());
}
#[test]
fn malformed_actions_and_scenarios_are_rejected_without_advancing() {
    let mut w = World::new(Scenario::default(), 0).unwrap();
    assert!(w.step(&[]).is_err());
    assert!(
        w.step(&[Action {
            throttle: f32::NAN,
            steering: 0.
        }])
        .is_err()
    );
    assert_eq!(w.steps, 0);
    assert!(Scenario::from_json(r#"{"physics_hz":121}"#).is_err());
    assert!(Scenario::from_json(r#"{"typo":true}"#).is_err());
}
#[test]
fn car_crosses_tile_seam_without_height_jump() {
    let mut cfg = Scenario {
        spawn_jitter: 0.,
        terrain: TerrainKind::Hills,
        ..Default::default()
    };
    cfg.vehicles[0].spawn = [-0.2, 0.];
    let mut w = World::new(cfg, 0).unwrap();
    drive(
        &mut w,
        70,
        Action {
            throttle: 0.5,
            steering: 0.25,
        },
    );
    let v = &w.snapshot().vehicles[0];
    assert!(v.position[0] > 0.);
    assert!(v.position[1] - w.terrain.height(v.position[0], v.position[2]) > 0.4);
    assert!(v.position[1] - w.terrain.height(v.position[0], v.position[2]) < 1.2);
}

#[test]
fn configurable_mass_changes_acceleration() {
    let cfg = Scenario {
        spawn_jitter: 0.,
        ..Default::default()
    };
    let mut light = World::new(cfg.clone(), 0).unwrap();
    let mut heavy_cfg = cfg;
    heavy_cfg.vehicles[0].parameters = Some(VehicleSpec {
        mass: 1800.,
        ..Default::default()
    });
    let mut heavy = World::new(heavy_cfg, 0).unwrap();
    let action = Action {
        throttle: 0.7,
        steering: 0.,
    };
    drive(&mut light, 40, action);
    drive(&mut heavy, 40, action);
    assert!(light.snapshot().vehicles[0].speed > heavy.snapshot().vehicles[0].speed * 1.5);
}
