/*!
[`Network`]: crate::network::Network
[`EdgeListEdge`]: crate::network::EdgeListEdge
[`NodeEdge`]: crate::network::NodeEdge
[`new`]: crate::shared::NetworkAnalysis::new
[`solve`]: crate::shared::NetworkAnalysis::solve
[`MeshAnalysis`]: crate::mesh_analysis::MeshAnalysis
[`NodalAnalysis`]: crate::nodal_analysis::NodalAnalysis
[`CurrentSources`]: crate::shared::CurrentSources
[`VoltageSources`]: crate::shared::VoltageSources
[`Resistances`]: crate::shared::Resistances
[`JacobianData`]: crate::shared::JacobianData
[`Solution`]: crate::shared::Solution
[`SolverConfig`]: crate::shared::SolverConfig
[`Type`]: crate::network::Type

A solver for nonlinear networks based on the mesh and nodal analysis methods.

 */
#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("example.svg", "docs/img/example.svg"),
doc = ::embed_doc_image::embed_image!("graph_terminology.svg", "docs/img/graph_terminology.svg"),
))]
#![cfg_attr(
    not(feature = "doc-images"),
    doc = "**Doc images not enabled**. Compile docs with `cargo doc --features 'doc-images'` and Rust version >= 1.54."
)]
#![doc = include_str!("../docs/main.md")]
#![deny(missing_docs)]

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
