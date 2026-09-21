use crate::TerrainKind;
use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

pub const SPACING: f32 = 4.;
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Material {
    pub id: u32,
    pub grip: f32,
    pub rolling: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TerrainMesh {
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
}
#[derive(Clone, Debug)]
pub struct Terrain {
    pub kind: TerrainKind,
    pub mixed: bool,
}
impl Terrain {
    fn sample(&self, x: f32, z: f32) -> f32 {
        match self.kind {
            TerrainKind::Flat => 0.,
            TerrainKind::Hills => 0.025 * z + 0.65 * (z * 0.09).sin() + 0.25 * (x * 0.14).sin(),
        }
    }
    /// Same diagonal and piecewise planar interpolation as the collision mesh.
    pub fn height(&self, x: f32, z: f32) -> f32 {
        let x0 = (x / SPACING).floor() * SPACING;
        let z0 = (z / SPACING).floor() * SPACING;
        let u = (x - x0) / SPACING;
        let v = (z - z0) / SPACING;
        let a = self.sample(x0, z0);
        let b = self.sample(x0 + SPACING, z0);
        let c = self.sample(x0, z0 + SPACING);
        let d = self.sample(x0 + SPACING, z0 + SPACING);
        if u + v <= 1. {
            a + (b - a) * u + (c - a) * v
        } else {
            d + (c - d) * (1. - u) + (b - d) * (1. - v)
        }
    }
    pub fn material(&self, x: f32, z: f32) -> Material {
        if self.mixed && x >= 0. && (30.0..50.).contains(&z) {
            Material {
                id: 2,
                grip: 0.12,
                rolling: 0.008,
            }
        } else if self.kind == TerrainKind::Hills {
            Material {
                id: 1,
                grip: 0.65,
                rolling: 0.035,
            }
        } else {
            Material {
                id: 0,
                grip: 1.1,
                rolling: 0.015,
            }
        }
    }
    pub fn meshes(&self) -> Vec<TerrainMesh> {
        [-32., 0.]
            .into_iter()
            .map(|x0| {
                let mut vertices = Vec::new();
                let mut triangles = Vec::new();
                for z in 0..=32 {
                    for x in 0..=8 {
                        let px = x0 + x as f32 * SPACING;
                        let pz = -16. + z as f32 * SPACING;
                        vertices.push([px, self.sample(px, pz), pz]);
                    }
                }
                for z in 0..32 {
                    for x in 0..8 {
                        let a = z * 9 + x;
                        triangles.push([a, a + 9, a + 1]);
                        triangles.push([a + 1, a + 9, a + 10]);
                    }
                }
                TerrainMesh {
                    vertices,
                    triangles,
                }
            })
            .collect()
    }
    pub fn insert(&self, colliders: &mut ColliderSet) {
        for (i, m) in self.meshes().into_iter().enumerate() {
            let vertices = m.vertices.into_iter().map(Point::from).collect();
            colliders.insert(
                ColliderBuilder::trimesh(vertices, m.triangles)
                    .expect("valid generated terrain")
                    .friction(0.8)
                    .user_data(i as u128 + 1),
            );
        }
        // A visible, physical obstacle alongside the route, also seen by range sensors.
        colliders.insert(
            ColliderBuilder::cuboid(1.5, 1., 1.5)
                .translation(vector![9., self.height(9., 45.) + 1., 45.])
                .user_data(100),
        );
    }
}
