/*!
Implementation of mesh analysis for [`Network`]s.

This module revolves around the [`MeshAnalysis`] struct which implements [`crate::base::NetworkAnalysis`]. See the struct documentation for details.
*/

use na::DMatrix;
use petgraph::visit::EdgeRef;

use crate::{network::*, shared::*};

/**
Implementation of the mesh analysis algorithm for a nonlinear [`Network`].

# Overview

This struct holds components (mainly conversion matrices and buffers) needed to perform a
[mesh analysis](<https://en.wikipedia.org/wiki/Mesh_analysis>) of a [`Network`]. The aforementioned conversion matrices are derived within
the [`NetworkAnalysis::new`] method. They are then used within the [`NetworkAnalysis::solve`] method
to try and solve the problem defined by the network, its excitation and resistances. An in-depth description of the solving process is given
in [lib module docstring](`crate::lib`).

To get an general overview over mesh analysis, please have a look at the corresponding [Wikipedia entry]((<https://en.wikipedia.org/wiki/Mesh_analysis>).
Some advanced features such as e.g. custom Jacobians require an in-depth understanding of the method. It is therefore recommended to consult specialist
literature such as \[1\].

In comparison to the closely related [nodal analysis](<https://en.wikipedia.org/wiki/Nodal_analysis>) method, which is available via the [`NodalAnalysis`]
struct, mesh analysis is especially well suited for networks with many nodes and few loops (i.e. many elements in serial) or with many voltage sources.
The number of equations is equal to the number of meshes ([`MeshAnalysis::mesh_count`]).

# Examples

The docstring of [`NetworkAnalysis::solve`] as well as the [lib module docstring](`crate::lib`) show some examples
on how to perform mesh analysis. Furthermore, a variety of examples is provided in the `examples` directory of the repository.

# Literature

1) Schmidt, Lorenz-Peter; Schaller, Gerd; Martius, Siegfried: Grundlagen der Elektrotechnik 3 - Netzwerke. 1st edition (2006). Pearson, Munich
 */
#[derive(Debug, Clone)]
pub struct MeshAnalysis {
    network_excitation: NetworkExcitation,
    coefficient_matrix: CoefficientMatrix,

    // Coupling matrices
    edge_to_mesh: DMatrix<f64>, // Each row corresponds to a mesh
    unknowns_to_edge_currents: DMatrix<f64>,
    edge_types: Vec<Type>,

    // Buffer storage
    buf: Buffer,
}

impl NetworkAnalysis for MeshAnalysis {
    fn new(network: &Network) -> Self {
        let (edge_to_mesh, number_current_source_meshes) = coupling_edge_to_mesh(&network);

        let unknowns_to_edge_currents = coupling_unknowns_to_edge_currents(&network, &edge_to_mesh);
        let edge_to_mesh_resistance =
            coupling_edge_to_mesh_resistance(network, &edge_to_mesh, number_current_source_meshes);
        let edge_to_mesh_excitation =
            coupling_excitation_edge_to_mesh(&network, &edge_to_mesh, number_current_source_meshes);

        let mesh_count = edge_to_mesh.nrows();
        let edge_count = edge_to_mesh.ncols();

        // Add a 1.0 on the main diagonal of every current source mesh. This value will not be changed later.
        let mut coeff_mat = DMatrix::repeat(mesh_count, mesh_count, 0.0);
        for mesh in 0..number_current_source_meshes {
            coeff_mat[(mesh, mesh)] = 1.0;
        }

        let mut edge_types = Vec::with_capacity(network.graph().edge_count());
        for weight in network.graph().edge_weights().cloned() {
            edge_types.push(weight);
        }

        return MeshAnalysis {
            edge_types,
            network_excitation: NetworkExcitation::new(edge_to_mesh_excitation),
            coefficient_matrix: CoefficientMatrix::new(coeff_mat, edge_to_mesh_resistance),
            edge_to_mesh,
            unknowns_to_edge_currents,
            buf: Buffer::new(edge_count, mesh_count),
        };
    }

    fn solve<'a>(
        &'a mut self,
        resistances: Resistances,
        current_exc: CurrentSources,
        voltage_src: VoltageSources,
        initial_edge_resistances: Option<&[f64]>,
        initial_edge_currents: Option<&[f64]>,
        jacobian: Option<&mut (dyn for<'b> FnMut(JacobianData<'b>) + 'a)>,
        config: &SolverConfig,
    ) -> Result<Solution<'a>, SolveError> {
        return NetworkAnalysisPriv::solve(
            self,
            resistances,
            current_exc,
            voltage_src,
            initial_edge_resistances,
            initial_edge_currents,
            jacobian,
            config,
        );
    }

    fn edge_types(&self) -> &[Type] {
        return self.edge_types.as_slice();
    }
}

impl NetworkAnalysisPriv for MeshAnalysis {
    fn calculate_edge_currents_and_voltages(&mut self) {
        // Edge currents
        self.buf
            .edge_currents
            .gemm(1.0, &self.unknowns_to_edge_currents, &self.buf.x, 0.0);

        // ===================================================================
        // Edge voltages

        // Calculate the voltage drop over the resistances via the edge currents and add the voltage sources.
        self.buf
            .edge_voltages
            .iter_mut()
            .zip(self.buf.edge_resistances.as_slice().iter())
            .zip(self.buf.edge_currents.as_slice().iter())
            .zip(self.network_excitation.edges().iter())
            .zip(self.edge_types.iter())
            .for_each(|((((voltage, resistance), current), exc), etype)| {
                if *etype == Type::Voltage {
                    *voltage = -*exc;
                } else {
                    *voltage = *resistance * *current;
                }
            });

        // Current sources
        // Iterate through all meshes / rows and add up the voltages (they should result in 0). If they don't,
        // a current source must be within the mesh which compensates the nonzero voltage.
        for mesh in self.edge_to_mesh.row_iter() {
            // Index of the current source of the mesh (if it has any current source)
            let mut current_source_and_dir: Option<(usize, f64)> = None;
            let mut voltage_sum = 0.0;
            for (idx, ((coupling, edge_type), voltage)) in mesh
                .iter()
                .cloned()
                .zip(self.edge_types().iter().cloned())
                .zip(self.buf.edge_voltages.iter().cloned())
                .enumerate()
            {
                // Each mesh can have at most one current source (checked during network construction).
                // Hence, this if clause is either fulfilled once per loop or not at all.
                if edge_type == Type::Current {
                    if coupling != 0.0 {
                        current_source_and_dir = Some((idx, coupling));
                    }
                } else {
                    voltage_sum += coupling * voltage;
                }
            }

            if let Some((idx, coupling)) = current_source_and_dir {
                // SAFETY: idx is generated from iterating over edge_voltages and is therefore always valid
                let v_ref = unsafe { self.buf.edge_voltages.get_unchecked_mut(idx) };

                // Compensate the nonzero voltage via the current source
                *v_ref = -coupling * voltage_sum;
            }
        }
    }

    fn split_mut<'a>(
        &'a mut self,
    ) -> (
        &'a mut NetworkExcitation,
        &'a mut CoefficientMatrix,
        &'a DMatrix<f64>,
        &'a [Type],
        &'a mut Buffer,
    ) {
        return (
            &mut self.network_excitation,
            &mut self.coefficient_matrix,
            &self.edge_to_mesh,
            self.edge_types.as_slice(),
            &mut self.buf,
        );
    }

    fn combine_resistance_and_coupling(edge: f64, coupling: f64) -> f64 {
        coupling * edge
    }
}

impl MeshAnalysis {
    /**
    Returns a matrix describing the coupling between the meshes and the equation system.

    The mesh analysis method derives `m` system equations from the input matrix, where
    `m` is the number of meshes.
    Together with the `n` edges of the underlying [`Network`], this results in a matrix `m x n`
    which directly describes the coupling between meshes and edges:
    * -1: Mesh direction is opposite to edge direction (as defined via `source -> target`).
    * 0: Edge is not part of the mesh.
    * 1: Mesh direction corresponds to edge direction.

    Therefore, this matrix together with [`MeshAnalysis::edge_types`] describes the entire equation system of the mesh analysis.
     */
    pub fn edge_to_mesh(&self) -> &DMatrix<f64> {
        return &self.edge_to_mesh;
    }

    /**
    Returns a conversion matrix for calculating the mesh resistance matrix from the edge resistances.

    This conversion matrix allows calculating the system matrix (the `A` in `A * x = b`). Each element
    of the conversion matrix is a vector whose length is either zero (if the mesh with the same index as the column
    has a current source) or equal to that of the edge resistance vector otherwise.
    Pairwise multiplication of this vector with the edge resistance vector and summing the resulting vector
    up returns the value of the corresponding system matrix element.

    This means that the system matrix columns for meshes containing a current source contains only zeros.
    For further explanation, see \[1\], p. 62ff.

    # Examples

    ```
    use network_analysis::*;
    use nalgebra::Matrix2;

    /*
    This creates the following network with a voltage source at 0
     ┌─[1]─┬─[2]─┐
    [0]   [6]   [3]
     └─[5]─┴─[4]─┘
     */
    let mut edges: Vec<EdgeListEdge> = Vec::new();
    edges.push(EdgeListEdge::new(vec![5], vec![1], Type::Voltage));
    edges.push(EdgeListEdge::new(vec![0], vec![2, 6], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![1, 6], vec![3], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![2], vec![4], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![3], vec![5, 6], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![4, 6], vec![0], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![1, 2], vec![4, 5], Type::Resistance));
    let network = Network::from_edge_list_edges(&edges).expect("valid network");

    /*
    This network forms two meshes -> The conversion matrix is 2x2 and each element
    is a vector of length 7 (since the matrix has seven edges)
     */
    let mesh_analysis = MeshAnalysis::new(&network);
    let conv = mesh_analysis.edge_to_mesh_resistance();
    assert_eq!(conv.nrows(), 2);
    assert_eq!(conv.ncols(), 2);
    for elem in conv.iter() {
        assert_eq!(elem.len(), 7);
    }

    // Use the conversion matrix to calculate the system matrix
    let edge_resistances = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let mut system_matrix = Matrix2::from_element(0.0);
    for (sys_elem, conv_vec) in system_matrix.iter_mut().zip(conv.iter()) {
        *sys_elem = conv_vec.iter().zip(edge_resistances.into_iter()).map(|(factor, edge_resistance)| factor * edge_resistance).sum();
    }

    // No current sources - matrix is symmetrical
    assert_eq!(system_matrix[(0, 0)], 3.0);
    assert_eq!(system_matrix[(0, 1)], -1.0);
    assert_eq!(system_matrix[(1, 0)], -1.0);
    assert_eq!(system_matrix[(1, 1)], 4.0);
    ```

    # Literature

    1) Schmidt, Lorenz-Peter; Schaller, Gerd; Martius, Siegfried: Grundlagen der Elektrotechnik 3 - Netzwerke. 1st edition (2006). Pearson, Munich
     */
    pub fn edge_to_mesh_resistance(&self) -> &DMatrix<Vec<f64>> {
        return &self.coefficient_matrix.edge_to_network_resistance();
    }

    /**
    Returns a conversion matrix from mesh to edge currents.
    This function is mainly meant to be used in custom Jacobian implementations.

    As explained in the docstring of [`JacobianFunctionSignature`], a custom Jacobian
    function receives the mesh currents as an input argument. The matrix provided by
    this function can then be used to calculate the edge currents via matrix
    multiplication:

    `C * m = e`, where `C` is this matrix, `m` is the mesh
    current vector and `e` is the edge current vector.
     */
    pub fn unknowns_to_edge_currents(&self) -> &DMatrix<f64> {
        return &self.unknowns_to_edge_currents;
    }

    /**
    Returns the number of meshes.

    # Examples

    ```
    use network_analysis::*;

    /*
    This creates the following network with a voltage source at 0
     ┌─[1]─┬─[2]─┐
    [0]   [6]   [3]
     └─[5]─┴─[4]─┘
     */
    let mut edges: Vec<EdgeListEdge> = Vec::new();
    edges.push(EdgeListEdge::new(vec![5], vec![1], Type::Voltage));
    edges.push(EdgeListEdge::new(vec![0], vec![2, 6], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![1, 6], vec![3], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![2], vec![4], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![3], vec![5, 6], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![4, 6], vec![0], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![1, 2], vec![4, 5], Type::Resistance));
    let network = Network::from_edge_list_edges(&edges).expect("valid network");

    let mesh_analysis = MeshAnalysis::new(&network);
    assert_eq!(mesh_analysis.mesh_count(), 2);
    ```
     */
    pub fn mesh_count(&self) -> usize {
        return self.edge_to_mesh.nrows();
    }

    /**
    Returns the number of edges of the underlying network.

    # Examples

    ```
    use network_analysis::*;

    /*
    This creates the following network with a voltage source at 0
     ┌─[1]─┬─[2]─┐
    [0]   [6]   [3]
     └─[5]─┴─[4]─┘
     */
    let mut edges: Vec<EdgeListEdge> = Vec::new();
    edges.push(EdgeListEdge::new(vec![5], vec![1], Type::Voltage));
    edges.push(EdgeListEdge::new(vec![0], vec![2, 6], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![1, 6], vec![3], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![2], vec![4], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![3], vec![5, 6], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![4, 6], vec![0], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![1, 2], vec![4, 5], Type::Resistance));
    let network = Network::from_edge_list_edges(&edges).expect("valid network");

    let mesh_analysis = MeshAnalysis::new(&network);
    assert_eq!(mesh_analysis.edge_count(), 7);
    ```
     */
    pub fn edge_count(&self) -> usize {
        return self.edge_types.len();
    }
} // impl

/**
Determine the coupling between all edges of the given network and the meshes.

This is the single most important coupling matrix, because all other coupling
matrices (excitation, resistnaces, ...) can be derived from it. It has the
dimension m x n, where m is the number of meshes and n is the number of edges.
Each value in this matrix is either -1, 0 or 1:
* -1: The mesh direction opposes the edge direction (negative coupling)
* 0: The edge is not included in the mesh (no coupling)
* 1: Mesh and edge direction are identical (positive coupling)
If two meshes share an edge, their coupling is identical to the product of the mesh-edge-couplings.

The first `n` meshes are current source meshes, where `n` is the second value returned from this function.
The matrix elements are f64 since this reduces the need for type conversions.
 */
fn coupling_edge_to_mesh(network: &Network) -> (DMatrix<f64>, usize) {
    let spanning_tree = find_spanning_tree(network.graph());

    // Build the coupling matrix
    let edge_count = network.graph().edge_count();
    let mesh_count = spanning_tree.independent_branches.len();
    let mut edge_to_mesh: DMatrix<f64> = DMatrix::repeat(mesh_count, edge_count, 0.0);

    for (mesh, branch) in spanning_tree.independent_branches.iter().enumerate() {
        // Add the elements of the independent branch
        for (edge_index, positive_coupling_mesh_edge) in
            branch.edges.iter().zip(branch.coupling.iter())
        {
            // The direction of the first edge defines the mesh direction
            if *positive_coupling_mesh_edge {
                edge_to_mesh[(mesh, edge_index.index())] = 1.0
            } else {
                edge_to_mesh[(mesh, edge_index.index())] = -1.0
            }
        }

        // Add the elements of the mesh which are part of the spanning tree

        // Get all nodes when traversing the spanning tree from start to finish of the independent branch
        let (_, nodes) = petgraph::algo::astar(
            &spanning_tree.graph,
            branch.source,
            |finish| finish == branch.target,
            |_| 1.0,
            |_| 0.0,
        )
        .expect("The spanning tree should always have a connection between the branch ends");

        let mut previous_edge_index = *branch
            .edges
            .front()
            .expect("Has already at least one element");
        for win in nodes.windows(2) {
            let edge_index = spanning_tree
                .graph
                .find_edge(win[0], win[1])
                .expect("Both nodes should connect an edge");

            let (source, target) = unsafe {
                network
                    .graph()
                    .edge_endpoints(edge_index)
                    .unwrap_unchecked()
            };
            let (prev_source, prev_target) = unsafe {
                network
                    .graph()
                    .edge_endpoints(previous_edge_index)
                    .unwrap_unchecked()
            };

            // Derive the coupling from the independent branch
            edge_to_mesh[(mesh, edge_index.index())] =
                if edge_to_mesh[(mesh, previous_edge_index.index())] == 1.0 {
                    if prev_source == target { 1.0 } else { -1.0 }
                } else {
                    if prev_target == source { -1.0 } else { 1.0 }
                };

            // Update the previous values
            previous_edge_index = edge_index;
        }
    }

    return (edge_to_mesh, spanning_tree.number_current_source_meshes);
}

fn coupling_unknowns_to_edge_currents(_: &Network, edge_to_mesh: &DMatrix<f64>) -> DMatrix<f64> {
    return edge_to_mesh.transpose();
}

fn coupling_edge_to_mesh_resistance(
    network: &Network,
    edge_to_mesh: &DMatrix<f64>,
    number_current_source_meshes: usize,
) -> DMatrix<Vec<f64>> {
    let mesh_count = edge_to_mesh.nrows();
    let edge_count = edge_to_mesh.ncols();

    /*
    Preallocate the coupling matrix. It has the size (row,col,depth), where
    the number of columns and the number of rows is equal to that of the mesh_matrix.
    The depth equals the number of edges.
    */
    let mut edge_to_mesh_resistance = DMatrix::repeat(mesh_count, mesh_count, Vec::new());

    // Current source meshes don't need to be populated. Hence, populating the meshes starts at "number_current_source_meshes"
    for mesh in number_current_source_meshes..mesh_count {
        // Add the edge vector for this entire row
        for ii in 0..mesh_count {
            edge_to_mesh_resistance[(mesh, ii)] = vec![0.0; edge_count];
        }

        // Iterate through all edges. For each edge, check if it is included in the current mesh and in all other meshes.
        for other_mesh in 0..mesh_count {
            let edge_vec = &mut edge_to_mesh_resistance[(mesh, other_mesh)];
            for (edge_idx, (edge, edge_type)) in edge_vec
                .iter_mut()
                .zip(network.graph().edge_weights())
                .enumerate()
            {
                if *edge_type == Type::Resistance {
                    *edge = edge_to_mesh[(mesh, edge_idx)] * edge_to_mesh[(other_mesh, edge_idx)];
                }
            }
        }
    }

    return edge_to_mesh_resistance;
}

fn coupling_excitation_edge_to_mesh(
    network: &Network,
    edge_to_mesh: &DMatrix<f64>,
    number_current_source_meshes: usize,
) -> DMatrix<f64> {
    let mesh_count = edge_to_mesh.nrows();
    let edge_count = edge_to_mesh.ncols();

    let mut excitation_edge_to_mesh: DMatrix<f64> = DMatrix::repeat(mesh_count, edge_count, 0.0);

    for edge in network.graph().edge_references() {
        // The first `number_current_source_meshes` elements of the excitation matrix includes current excitations.
        // However, all current sources which are not on an independent branch are ignored.
        let edge_idx = edge.id().index();
        if edge.weight() == &Type::Current && edge_is_on_independent_branch(&edge_to_mesh, edge_idx)
        {
            for mesh in 0..number_current_source_meshes {
                excitation_edge_to_mesh[(mesh, edge_idx)] = edge_to_mesh[(mesh, edge_idx)] as f64;
            }
        // Populate the coupling matrix with the voltage source vector
        } else if edge.weight() == &Type::Voltage {
            for mesh in number_current_source_meshes..mesh_count {
                excitation_edge_to_mesh[(mesh, edge_idx)] = edge_to_mesh[(mesh, edge_idx)] as f64;
            }
        }
    }
    return excitation_edge_to_mesh;
}

fn edge_is_on_independent_branch(edge_to_mesh: &DMatrix<f64>, edge: usize) -> bool {
    let mut occurences = 0;
    for mesh in 0..edge_to_mesh.nrows() {
        if edge_to_mesh[(mesh, edge)] != 0.0 {
            occurences += 1;
        }
    }
    return occurences < 2;
}

#[cfg(test)]
mod tests {

    use super::*;

    /// Example network from the README.md (image see doc/example.svg)
    /// The excitations can be adjusted via the `exc` input parameter.
    fn network_creation(first_exc_is_voltage: bool, second_exc_is_voltage: bool) -> Network {
        let first_edge = if first_exc_is_voltage {
            Type::Voltage
        } else {
            Type::Current
        };
        let second_edge = if second_exc_is_voltage {
            Type::Voltage
        } else {
            Type::Current
        };

        let edges = [
            EdgeListEdge::new(vec![1], vec![2, 3], first_edge),
            EdgeListEdge::new(vec![2, 4, 5], vec![0], Type::Resistance),
            EdgeListEdge::new(vec![0, 3], vec![1, 4, 5], Type::Resistance),
            EdgeListEdge::new(vec![0, 2], vec![4, 6], Type::Resistance),
            EdgeListEdge::new(vec![3, 6], vec![1, 2, 5], Type::Resistance),
            EdgeListEdge::new(vec![1, 2, 4], vec![6], second_edge),
            EdgeListEdge::new(vec![5], vec![3, 4], Type::Resistance),
        ];
        return Network::from_edge_list_edges(edges.as_slice()).expect("this is a valid network");
    }

    #[test]
    fn test_coupling_edge_to_mesh() {
        {
            let network = network_creation(true, true);

            let (m, number_current_sources) = coupling_edge_to_mesh(&network);
            assert_eq!(number_current_sources, 0);
            assert_eq!(m.ncols(), network.graph().edge_count());
            assert_eq!(m.nrows(), 3);

            let r0: Vec<i32> = m.row(0).iter().map(|v| *v as i32).collect();
            let r1: Vec<i32> = m.row(1).iter().map(|v| *v as i32).collect();
            let r2: Vec<i32> = m.row(2).iter().map(|v| *v as i32).collect();

            assert_eq!(r0, vec![1, 1, 0, 1, 0, -1, -1]);
            assert_eq!(r1, vec![0, 0, 1, -1, 0, 1, 1]);
            assert_eq!(r2, vec![0, 0, 0, 0, 1, 1, 1]);
        }

        {
            let network = network_creation(true, false);

            let (m, number_current_sources) = coupling_edge_to_mesh(&network);
            assert_eq!(number_current_sources, 1);
            assert_eq!(m.ncols(), network.graph().edge_count());
            assert_eq!(m.nrows(), 3);

            let r0: Vec<i32> = m.row(0).iter().map(|v| *v as i32).collect();
            let r1: Vec<i32> = m.row(1).iter().map(|v| *v as i32).collect();
            let r2: Vec<i32> = m.row(2).iter().map(|v| *v as i32).collect();

            assert_eq!(r0, vec![0, 0, 0, 0, 1, 1, 1]);
            assert_eq!(r1, vec![1, 1, 0, 1, 1, 0, 0]);
            assert_eq!(r2, vec![0, 0, 1, -1, -1, 0, 0]);
        }

        {
            let network = network_creation(false, false);

            let (m, number_current_sources) = coupling_edge_to_mesh(&network);
            assert_eq!(number_current_sources, 2);
            assert_eq!(m.ncols(), network.graph().edge_count());
            assert_eq!(m.nrows(), 3);

            let r0: Vec<i32> = m.row(0).iter().map(|v| *v as i32).collect();
            let r1: Vec<i32> = m.row(1).iter().map(|v| *v as i32).collect();
            let r2: Vec<i32> = m.row(2).iter().map(|v| *v as i32).collect();

            assert_eq!(r0, vec![1, 1, 0, 1, 1, 0, 0]);
            assert_eq!(r1, vec![0, 0, 0, 0, 1, 1, 1]);
            assert_eq!(r2, vec![0, 0, 1, -1, -1, 0, 0]);
        }

        {
            let network = network_creation(false, true);

            let (m, number_current_sources) = coupling_edge_to_mesh(&network);
            assert_eq!(number_current_sources, 1);
            assert_eq!(m.ncols(), network.graph().edge_count());
            assert_eq!(m.nrows(), 3);

            let r0: Vec<i32> = m.row(0).iter().map(|v| *v as i32).collect();
            let r1: Vec<i32> = m.row(1).iter().map(|v| *v as i32).collect();
            let r2: Vec<i32> = m.row(2).iter().map(|v| *v as i32).collect();

            assert_eq!(r0, vec![1, 1, 0, 1, 0, -1, -1]);
            assert_eq!(r1, vec![0, 0, 1, -1, 0, 1, 1]);
            assert_eq!(r2, vec![0, 0, 0, 0, 1, 1, 1]);
        }
    }

    #[test]
    fn test_coupling_edge_to_mesh_resistance() {
        {
            let network = network_creation(true, true);

            let (m, n) = coupling_edge_to_mesh(&network);
            let c = coupling_edge_to_mesh_resistance(&network, &m, n);
            assert_eq!(c.ncols(), 3);
            assert_eq!(c.nrows(), 3);

            // The diagonals hold the meshes themselves
            assert_eq!(c[(0, 0)], vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
            assert_eq!(c[(1, 1)], vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
            assert_eq!(c[(2, 2)], vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
        }
    }
}
