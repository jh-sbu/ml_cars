use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use ml_cars_controllers::{Recording, Runner};
use ml_cars_sim::{Scenario, Snapshot, Terrain, route_x};
use std::{env, fs};
struct Session {
    runner: Runner,
    replay: Option<Recording>,
    frame: usize,
    paused: bool,
    accumulator: f32,
    snapshot: Snapshot,
    error: Option<String>,
}
#[derive(Component)]
struct Car(usize);
#[derive(Component)]
struct Wheel(usize, usize);
#[derive(Component)]
struct Hud;
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().collect();
    let replay = if args.get(1).is_some_and(|s| s == "--replay") {
        Some(Recording::load(
            args.get(2).ok_or("--replay needs a recording path")?,
        )?)
    } else {
        None
    };
    let scenario = if let Some(r) = &replay {
        r.scenario.clone()
    } else {
        Scenario::from_json(
            &fs::read_to_string(
                args.get(1)
                    .map(String::as_str)
                    .unwrap_or("scenarios/traffic.json"),
            )
            .map_err(|e| e.to_string())?,
        )?
    };
    let seed = replay.as_ref().map(|r| r.seed).unwrap_or(42);
    // Replay needs no deployed model files: it consumes recorded poses only.
    let mut runtime_scenario = scenario.clone();
    if replay.is_some() {
        for v in &mut runtime_scenario.vehicles {
            v.controller = "native".into();
        }
    }
    let runner = Runner::new(runtime_scenario, seed)?;
    let snapshot = replay
        .as_ref()
        .map(|r| r.frames[0].clone())
        .unwrap_or_else(|| runner.world.snapshot());
    App::new()
        .insert_non_send_resource(Session {
            runner,
            replay,
            frame: 0,
            paused: false,
            accumulator: 0.,
            snapshot,
            error: None,
        })
        .insert_resource(ClearColor(Color::srgb(0.055, 0.075, 0.11)))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "ML Cars | Space pause · N step · R reset · Tab follow".into(),
                resolution: (1280, 800).into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(Update, (advance, draw).chain())
        .add_systems(Update, smoke_capture)
        .run();
    Ok(())
}
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    session: NonSend<Session>,
) {
    let terrain = &session.runner.world.terrain;
    for source in terrain.meshes() {
        let colors: Vec<[f32; 4]> = source
            .vertices
            .iter()
            .map(|p| match terrain.material(p[0], p[2]).id {
                2 => [0.45, 0.72, 0.86, 1.],
                1 => [0.39, 0.30, 0.18, 1.],
                _ => [0.21, 0.31, 0.26, 1.],
            })
            .collect();
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, source.vertices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(
            source.triangles.into_iter().flatten().collect(),
        ));
        mesh.compute_normals();
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(StandardMaterial {
                perceptual_roughness: 1.,
                ..default()
            })),
        ));
    }
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(3., 2., 3.))),
        MeshMaterial3d(materials.add(Color::srgb(0.85, 0.35, 0.12))),
        Transform::from_xyz(9., terrain.height(9., 45.) + 1., 45.),
    ));
    for (i, v) in session.runner.world.scenario.vehicles.iter().enumerate() {
        let s = v.spec();
        let color = if v.aware {
            Color::srgb(0.1, 0.7, 0.95)
        } else {
            Color::srgb(0.95, 0.6, 0.15)
        };
        commands.spawn((
            Car(i),
            Mesh3d(meshes.add(Cuboid::new(s.width, 0.5, s.length))),
            MeshMaterial3d(materials.add(color)),
            Transform::default(),
        ));
        for j in 0..4 {
            commands.spawn((
                Wheel(i, j),
                Mesh3d(meshes.add(Sphere::new(0.32))),
                MeshMaterial3d(materials.add(Color::srgb(0.06, 0.07, 0.09))),
                Transform::default(),
            ));
        }
    }
    commands.spawn((
        DirectionalLight {
            illuminance: 12000.,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(-20., 40., 20.).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(25., 30., -25.).looking_at(Vec3::new(0., 0., 30.), Vec3::Y),
    ));
    commands.spawn((
        Hud,
        Text::new(""),
        TextFont {
            font_size: 20.,
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            top: px(18),
            left: px(18),
            ..default()
        },
    ));
}
fn advance(mut session: NonSendMut<Session>, keys: Res<ButtonInput<KeyCode>>, time: Res<Time>) {
    if keys.just_pressed(KeyCode::Space) {
        session.paused = !session.paused;
        session.accumulator = 0.;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        session.frame = 0;
        session.accumulator = 0.;
        session.error = None;
        if let Some(r) = &session.replay {
            session.snapshot = r.frames[0].clone();
        } else {
            let seed = session.runner.world.seed;
            if let Err(e) = session.runner.reset(seed) {
                session.error = Some(e);
            }
            session.snapshot = session.runner.world.snapshot();
        }
    }
    let dt = 1. / session.runner.world.scenario.control_hz as f32;
    if !session.paused {
        session.accumulator += time.delta_secs().min(0.25);
    }
    let single = keys.just_pressed(KeyCode::KeyN);
    let steps = if single {
        session.paused = true;
        session.accumulator = 0.;
        1
    } else {
        (session.accumulator / dt).floor() as usize
    };
    for _ in 0..steps {
        if !single {
            session.accumulator -= dt;
        }
        if session.replay.is_some() {
            let next = session.frame + 1;
            let r = session.replay.as_ref().unwrap();
            if next < r.frames.len() {
                session.snapshot = r.frames[next].clone();
                session.frame = next;
            } else {
                session.paused = true;
                session.accumulator = 0.;
                break;
            }
        } else if session.runner.world.active().iter().any(|x| *x) {
            let n = session.runner.world.scenario.vehicles.len();
            if let Err(e) = session.runner.step(&vec![None; n]) {
                session.error = Some(e);
                session.paused = true;
                break;
            }
            session.snapshot = session.runner.world.snapshot();
        } else {
            session.paused = true;
            session.accumulator = 0.;
            break;
        }
    }
}
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn draw(
    session: NonSend<Session>,
    mut cars: Query<(&Car, &mut Transform)>,
    mut wheels: Query<(&Wheel, &mut Transform), Without<Car>>,
    mut cameras: Query<&mut Transform, (With<Camera3d>, Without<Car>, Without<Wheel>)>,
    mut hud: Query<&mut Text, With<Hud>>,
    mut gizmos: Gizmos,
    keys: Res<ButtonInput<KeyCode>>,
    mut follow: Local<usize>,
) {
    let state = &session.snapshot;
    if keys.just_pressed(KeyCode::Tab) {
        *follow = (*follow + 1) % (state.vehicles.len() + 1);
    }
    for (car, mut transform) in &mut cars {
        let v = &state.vehicles[car.0];
        transform.translation = Vec3::from_array(v.position);
        transform.rotation = Quat::from_array(v.rotation);
    }
    for (wheel, mut transform) in &mut wheels {
        if let Some(p) = state.vehicles[wheel.0].wheels.get(wheel.1) {
            transform.translation = Vec3::from_array(*p);
        }
    }
    if *follow > 0 {
        let p = Vec3::from_array(state.vehicles[*follow - 1].position);
        for mut t in &mut cameras {
            *t = Transform::from_translation(p + Vec3::new(12., 10., -15.))
                .looking_at(p + Vec3::Z * 5., Vec3::Y);
        }
    } else {
        for mut t in &mut cameras {
            *t = Transform::from_xyz(25., 30., -25.).looking_at(Vec3::new(0., 0., 35.), Vec3::Y);
        }
    }
    let terrain: &Terrain = &session.runner.world.terrain;
    for z in 0..90 {
        let p = |z: f32| Vec3::new(route_x(z), terrain.height(route_x(z), z) + 0.08, z);
        gizmos.line(p(z as f32), p(z as f32 + 1.), Color::srgb(0.9, 0.95, 0.65));
    }
    if session.replay.is_none() {
        let obs = session.runner.world.observations();
        for (i, v) in state.vehicles.iter().enumerate() {
            let p = Vec3::from_array(v.position);
            let q = Quat::from_array(v.rotation);
            let range = session.runner.world.scenario.vehicles[i].sensing_range;
            for (j, a) in [-0.8_f32, -0.4, 0., 0.4, 0.8].iter().enumerate() {
                gizmos.line(
                    p,
                    p + q * Vec3::new(a.sin(), 0., a.cos()) * range * obs[i][8 + j],
                    Color::srgba(0.3, 0.85, 1., 0.5),
                );
            }
        }
    }
    for mut t in &mut hud {
        **t = format!(
            "ML CARS  |  {}  |  t = {:.2}s\nSpace: pause   N: step   R: reset   Tab: follow\nCyan: aware   Amber: unaware   Blue ground: low grip\n{}{}",
            if session.replay.is_some() {
                "REPLAY"
            } else if session.paused {
                "PAUSED"
            } else {
                "LIVE"
            },
            state.time,
            state
                .vehicles
                .iter()
                .map(|v| format!(
                    "{}  {:.1} m/s  {}  materials {:?}",
                    v.id,
                    v.speed,
                    if v.active { "active" } else { "finished" },
                    v.materials
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            session
                .error
                .as_ref()
                .map(|e| format!("\nERROR: {e}"))
                .unwrap_or_default()
        );
    }
}

// Opt-in runtime validation; normal interactive sessions are unaffected.
fn smoke_capture(
    mut commands: Commands,
    mut frames: Local<u32>,
    mut exit: MessageWriter<bevy::app::AppExit>,
) {
    let Ok(path) = std::env::var("ML_CARS_SCREENSHOT") else {
        return;
    };
    *frames += 1;
    if *frames == 60 {
        commands
            .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
            .observe(bevy::render::view::screenshot::save_to_disk(path));
    }
    if *frames == 180 {
        exit.write(bevy::app::AppExit::Success);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn app(replay: Option<Recording>) -> App {
        let runner = Runner::new(Scenario::default(), 42).unwrap();
        let snapshot = runner.world.snapshot();
        let mut app = App::new();
        app.insert_non_send_resource(Session {
            runner,
            replay,
            frame: 0,
            paused: false,
            accumulator: 0.,
            snapshot,
            error: None,
        });
        app.insert_resource(Time::<()>::default());
        app.init_resource::<ButtonInput<KeyCode>>();
        app.add_systems(Update, advance);
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_millis(50));
        app
    }

    fn press(app: &mut App, key: KeyCode) {
        let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        keys.reset_all();
        keys.press(key);
    }

    #[test]
    fn pause_single_step_and_seeded_reset() {
        let mut app = app(None);
        app.update();
        assert_eq!(
            app.world()
                .non_send_resource::<Session>()
                .runner
                .world
                .steps,
            1
        );
        press(&mut app, KeyCode::Space);
        app.update();
        assert!(app.world().non_send_resource::<Session>().paused);
        assert_eq!(
            app.world()
                .non_send_resource::<Session>()
                .runner
                .world
                .steps,
            1
        );
        press(&mut app, KeyCode::KeyN);
        app.update();
        assert_eq!(
            app.world()
                .non_send_resource::<Session>()
                .runner
                .world
                .steps,
            2
        );
        press(&mut app, KeyCode::KeyR);
        app.update();
        assert_eq!(
            app.world()
                .non_send_resource::<Session>()
                .runner
                .world
                .steps,
            0
        );
        assert_eq!(app.world().non_send_resource::<Session>().snapshot.time, 0.);
    }

    #[test]
    fn playback_advances_stored_frames_without_stepping_physics() {
        let mut runner = Runner::new(Scenario::default(), 42).unwrap();
        let first = runner.world.snapshot();
        runner.step(&[None]).unwrap();
        let second = runner.world.snapshot();
        let expected = second.vehicles[0].position;
        let mut app = app(Some(Recording {
            version: 1,
            scenario: Scenario::default(),
            seed: 42,
            frames: vec![first, second],
        }));
        app.update();
        let session = app.world().non_send_resource::<Session>();
        assert_eq!(session.snapshot.vehicles[0].position, expected);
        assert_eq!(session.runner.world.steps, 0);
        app.update();
        assert!(app.world().non_send_resource::<Session>().paused);
        press(&mut app, KeyCode::KeyR);
        app.update();
        assert_eq!(app.world().non_send_resource::<Session>().frame, 0);
    }
}
