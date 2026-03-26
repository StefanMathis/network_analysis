#![doc = include_str!("../README.md")]

extern crate nalgebra as na;
extern crate petgraph;

mod finite_diff;
pub mod mesh_analysis;
pub mod network;
pub mod nodal_analysis;
pub mod shared;

pub use mesh_analysis::*;
pub use network::*;
pub use nodal_analysis::*;
pub use shared::*;
pub use std::marker::PhantomData;
