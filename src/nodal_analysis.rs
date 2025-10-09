/*!
Implementation of nodal analysis for [`Network`]s.

This module revolves around the [`NodalAnalysis`] struct which implements [`crate::base::NetworkAnalysis`]. See the struct documentation for details.
*/

use na::DMatrix;
use petgraph::visit::EdgeRef;

use crate::{network::*, shared::*};

/**
Implementation of the nodal analysis algorithm for a nonlinear [`Network`].

This struct holds components (mainly conversion matrices and buffers) needed to perform a modified
[nodal analysis](<https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA3.html>) of a [`Network`]. The aforementioned conversion matrices are derived within
the [`NetworkAnalysis::new`] method. They are then used within the [`NetworkAnalysis::solve`] method
to try and solve the problem defined by the network, its excitation and resistances. An in-depth description of the solving process is given
in [lib module docstring](`crate::lib`).

To get an general overview over nodal analysis, please have a look at the corresponding [Wikipedia entry]((<https://en.wikipedia.org/wiki/Nodal_analysis>).
Some advanced features such as e.g. voltage sources or using custom Jacobians require an in-depth understanding of the method. It is therefore recommended to consult specialist
literature such as \[1\], \[2\].

In comparison to the closely related [mesh analysis](https://en.wikipedia.org/wiki/Mesh_analysis) method, which is available via the [`MeshAnalysis`]
struct, nodal analysis is especially well suited for networks with a low number of nodes in comparison to the number of edges and few voltage sources
since the number of equations which need to be solved is equal to the number of nodes minus one (see [`NodalAnalysis::node_count()`]) plus the number of voltage sources.

# Examples

The docstring of [`NetworkAnalysis::solve`] as well as the [lib module docstring](`crate::lib`) show some examples
on how to perform nodal analysis. Furthermore, a variety of examples is provided in the `examples` directory of the repository.

# Literature

1) Schmidt, Lorenz-Peter; Schaller, Gerd; Martius, Siegfried: Grundlagen der Elektrotechnik 3 - Netzwerke. 1st edition (2006). Pearson, Munich
2) Modified nodal analysis: <https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA3.html>
 */
#[derive(Debug, Clone)]
pub struct NodalAnalysis {
    network_excitation: NetworkExcitation,
    coefficient_matrix: CoefficientMatrix,

    // Coupling matrices
    edge_to_node: DMatrix<f64>, // Each row corresponds to a node equation
    unknowns_to_edge_voltage: DMatrix<f64>,
    zero_potential_node: usize,
    edge_types: Vec<Type>,

    // Buffer storage
    buf: Buffer,
}

impl NetworkAnalysis for NodalAnalysis {
    fn new(network: &Network) -> Self {
        let (edge_to_node, zero_potential_node) = coupling_edge_to_node(network);
        let unknowns_to_edge_voltage = coupling_unknowns_to_edge_voltage(network, &edge_to_node);
        let edge_to_node_resistance = coupling_edge_to_node_resistance(network, &edge_to_node);
        let edge_to_node_exc = coupling_excitation_edge_to_node(network, &edge_to_node);

        let coeff_mat = initialize_system_matrix(network, &edge_to_node);

        let mut edge_types = Vec::with_capacity(network.graph().edge_count());
        for weight in network.graph().edge_weights().cloned() {
            edge_types.push(weight);
        }

        let number_node_equations = coeff_mat.ncols();

        return NodalAnalysis {
            network_excitation: NetworkExcitation::new(edge_to_node_exc),
            coefficient_matrix: CoefficientMatrix::new(coeff_mat, edge_to_node_resistance),
            edge_to_node,
            unknowns_to_edge_voltage,
            zero_potential_node,
            edge_types,
            buf: Buffer::new(network.graph().edge_count(), number_node_equations),
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

impl NetworkAnalysisPriv for NodalAnalysis {
    fn calculate_edge_currents_and_voltages(&mut self) {
        // Edge voltages
        self.buf
            .edge_voltages
            .gemm(1.0, &self.unknowns_to_edge_voltage, &self.buf.x, 0.0);

        // ===================================================================
        // Edge Currents

        // Calculate the currents over the resistances via the edge voltages and add the current sources.
        self.buf
            .edge_currents
            .iter_mut()
            .zip(self.buf.edge_resistances.as_slice().iter())
            .zip(self.buf.edge_voltages.as_slice().iter())
            .zip(self.network_excitation.edges().iter())
            .zip(self.edge_types.iter())
            .for_each(|((((current, resistance), voltage), exc), etype)| {
                if *etype == Type::Current {
                    *current = *exc;
                } else {
                    // Branchless algorithm to avoid dividing by zero
                    let corrected_resistance = (*resistance == 0.0) as u8 as f64 + *resistance;
                    *current = *voltage / corrected_resistance;
                }
            });

        // The currents over the voltage sources can be read out directly from the vector of unknowns.
        for (unknown_idx, (voltage_source_edge_idx, _)) in self
            .edge_types
            .iter()
            .cloned()
            .enumerate()
            .filter(|(_, edge_type)| *edge_type == Type::Voltage)
            .enumerate()
        {
            self.buf.edge_currents[voltage_source_edge_idx] =
                -self.buf.x[unknown_idx + self.node_count() - 1]
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
            &self.edge_to_node,
            self.edge_types.as_slice(),
            &mut self.buf,
        );
    }

    fn combine_resistance_and_coupling(edge: f64, coupling: f64) -> f64 {
        // Branchless algorithm to avoid dividing by zero
        let corrected_edge = (coupling == 0.0) as u8 as f64 + edge;
        coupling / corrected_edge
    }
}

impl NodalAnalysis {
    /**
    Returns a matrix describing the coupling between the edges and the equation system.

    The nodal analysis method derives `m` system equations from the input matrix, where
    `m` is the number of nodes minus one (one of the nodes is defined as potential 0,
    see [`NodalAnalysis::zero_potential_node`], hence its equation can be omitted).
    Together with the `n` edges of the underlying [`Network`], this results in a matrix `m x n`
    which directly describes the coupling between nodes and edges:
    * -1: Node is the source of the edge.
    * 0: Edge is not using the node.
    * 1: Node is the target of the edge.

    Therefore, this matrix together with [`NodalAnalysis::edge_types`] describes the entire equation system of the nodal analysis.
     */
    pub fn edge_to_node(&self) -> &DMatrix<f64> {
        return &self.edge_to_node;
    }

    /**
    Returns a conversion matrix for calculating the node resistance matrix from the edge resistances.

    This conversion matrix allows calculating the system matrix (the `A` in `A * x = b`). Each element
    of the conversion matrix is a vector whose length is equal to that of the edge resistance vector otherwise.
    Pairwise multiplication of this vector with the edge resistance vector and summing the resulting vector
    up returns the value of the corresponding system matrix element.

    # Examples

    ```
    use network_analysis::*;
    use nalgebra::Matrix5;

    /*
    This creates the following network with a current source at 0
     ┌─[1]─┬─[2]─┐
    [0]   [6]   [3]
     └─[5]─┴─[4]─┘
     */
    let mut edges: Vec<EdgeListEdge> = Vec::new();
    edges.push(EdgeListEdge::new(vec![5], vec![1], Type::Current));
    edges.push(EdgeListEdge::new(vec![0], vec![2, 6], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![1, 6], vec![3], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![2], vec![4], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![3], vec![5, 6], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![4, 6], vec![0], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![1, 2], vec![4, 5], Type::Resistance));
    let network = Network::from_edge_list_edges(&edges).expect("valid network");

    /*
    This network has 6 nodes -> The conversion matrix is 5x5 and each element
    is a vector of length 7 (since the matrix has seven edges)
     */
    let nodal_analysis = NodalAnalysis::new(&network);
    let conv = nodal_analysis.edge_to_node_resistance();
    assert_eq!(conv.nrows(), 5);
    assert_eq!(conv.ncols(), 5);
    for elem in conv.iter() {
        assert_eq!(elem.len(), 7);
    }

    // Use the conversion matrix to calculate the system matrix
    let edge_resistances = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let mut system_matrix = Matrix5::from_element(0.0);
    for (sys_elem, conv_vec) in system_matrix.iter_mut().zip(conv.iter()) {
        *sys_elem = conv_vec.iter().zip(edge_resistances.into_iter()).map(|(factor, edge_resistance)| factor / edge_resistance).sum();
    }

    let mut expected = Matrix5::from_element(0.0);

    // Diagonals
    expected[(0, 0)] = 1.0;
    expected[(1, 1)] = 1.0;
    expected[(2, 2)] = 2.0;
    expected[(3, 3)] = 2.0;
    expected[(4, 4)] = 3.0;

    // Off-diagonals
    expected[(0, 4)] = -1.0;
    expected[(4, 0)] = expected[(0, 4)];
    expected[(2, 3)] = -1.0;
    expected[(3, 2)] = expected[(2, 3)];
    expected[(3, 4)] = -1.0;
    expected[(4, 3)] = expected[(3, 4)];
    assert_eq!(system_matrix, expected);
    ```

    # Literature

    1) Schmidt, Lorenz-Peter; Schaller, Gerd; Martius, Siegfried: Grundlagen der Elektrotechnik 3 - Netzwerke. 1st edition (2006). Pearson, Munich
     */
    pub fn edge_to_node_resistance(&self) -> &DMatrix<Vec<f64>> {
        return self.coefficient_matrix.edge_to_network_resistance();
    }

    /**
    Returns a conversion matrix from node to edge voltages.
    This function is mainly meant to be used in custom Jacobian implementations.

    As explained in the docstring of [`JacobianFunctionSignature`], a custom Jacobian
    function receives the node voltages as an input argument. The matrix provided by
    this function can then be used to calculate the edge currents via matrix
    multiplication:

    `C * n = e`, where `C` is this matrix, `n` is the node
    voltage vector and `e` is the edge voltage vector.
     */
    pub fn unknowns_to_edge_voltage(&self) -> &DMatrix<f64> {
        return &self.unknowns_to_edge_voltage;
    }

    /**
    Returns the number of nodes in the network.

    The equation system size is equal to this number minus one.

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

    let nodal_analysis = NodalAnalysis::new(&network);
    assert_eq!(nodal_analysis.node_count(), 6);
    ```
     */
    pub fn node_count(&self) -> usize {
        return self.edge_to_node.nrows() + 1;
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

    let nodal_analysis = NodalAnalysis::new(&network);
    assert_eq!(nodal_analysis.edge_count(), 7);
    ```
     */
    pub fn edge_count(&self) -> usize {
        return self.edge_types.len();
    }

    /**
    Returns the index of the node which was chosen as the "zero potential" node.
    This node is always the one with the most edges using it, as this
    results in the most zeros in the system matrix of nodal analysis, hence reducing
    calculation load.
     */
    pub fn zero_potential_node(&self) -> usize {
        return self.zero_potential_node;
    }
}

/**
Determine the coupling between all edges of the given network and the nodes.

See [`NodalAnalysis::edge_to_node`]. The matrix elements are f64 since this removes the need for type conversions.
 */
fn coupling_edge_to_node(network: &Network) -> (DMatrix<f64>, usize) {
    // Find the node with the most edges
    let mut node_iterator = network.graph().node_indices();
    let mut zero_potential_node = node_iterator.next().expect("must have at least one node");
    let mut edge_count = network.graph().edges(zero_potential_node).count();
    for node in node_iterator {
        let ecount = network.graph().edges(node).count();
        if ecount > edge_count {
            edge_count = ecount;
            zero_potential_node = node;
        }
    }

    let mut edge_to_node: DMatrix<f64> = DMatrix::repeat(
        network.graph().node_count() - 1,
        network.graph().edge_count(),
        0.0,
    );

    // Iterate over all nodes again (skipping "zero_potential_node")
    for (row_index, node) in network
        .graph()
        .node_indices()
        .filter(|idx| *idx != zero_potential_node)
        .enumerate()
    {
        for edge in network.graph().edges(node) {
            let end_points = network
                .graph()
                .edge_endpoints(edge.id())
                .expect("edge exists");
            if end_points.0 == node {
                edge_to_node[(row_index, edge.id().index())] = 1.0;
            } else {
                edge_to_node[(row_index, edge.id().index())] = -1.0;
            }
        }
    }
    return (edge_to_node, zero_potential_node.index());
}

fn coupling_unknowns_to_edge_voltage(
    network: &Network,
    edge_to_node: &DMatrix<f64>,
) -> DMatrix<f64> {
    let mut coupling_matrix = DMatrix::repeat(
        edge_to_node.ncols(),
        network.voltage_source_count() + edge_to_node.nrows(),
        0.0,
    );

    // Transpose edge_to_node into the first columns of the matrix
    for row in 0..edge_to_node.ncols() {
        for col in 0..edge_to_node.nrows() {
            coupling_matrix[(row, col)] = -edge_to_node[(col, row)];
        }
    }
    return coupling_matrix;
}

/**
Calculates the "G" coupling matrix from https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA3.html

Each element of the conversion matrix is a vector whose length is equal to that of the edge resistance vector.
Pairwise division of this vector by the edge resistance vector and summing the resulting vector
up returns the value of the corresponding system matrix element.
 */
fn coupling_edge_to_node_resistance(
    network: &Network,
    edge_to_node: &DMatrix<f64>,
) -> DMatrix<Vec<f64>> {
    let node_equations = edge_to_node.nrows();
    let nedges = edge_to_node.ncols();
    let mut edge_to_node_resistance: DMatrix<Vec<f64>> =
        DMatrix::repeat(node_equations, node_equations, vec![0.0; nedges]);

    for row in 0..node_equations {
        for col in 0..node_equations {
            for (edge_idx, edge_type) in network.graph().edge_weights().cloned().enumerate() {
                // Check if the edge is a resistance and is coupled to the node(s)
                if edge_type == Type::Resistance
                    && edge_to_node[(row, edge_idx)] != 0.0
                    && edge_to_node[(col, edge_idx)] != 0.0
                {
                    if row == col {
                        // Diagonal
                        edge_to_node_resistance[(row, col)][edge_idx] = 1.0;
                    } else {
                        // Off-diagonal
                        edge_to_node_resistance[(row, col)][edge_idx] = -1.0;
                    }
                }
            }
        }
    }

    return edge_to_node_resistance;
}

/**
Calculates the "z" coupling matrix from https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA3.html

To get the actual values of "z", simply multiply this matrix with the excitation vector.
The excitation vector is simply the edge vector, where each source entry is the value of the source
and each resistance value is set to zero. For example, for a network with 7 elements and sources at 0 (value = 3.4)
and 5 (value = -1.5), the excitation vector is: [3.4; 0; 0; 0; 0; -1.5; 0].
 */
fn coupling_excitation_edge_to_node(
    network: &Network,
    edge_to_node: &DMatrix<f64>,
) -> DMatrix<f64> {
    let edge_count = network.graph().edge_count();
    let nodal_equations = network.graph().node_count() - 1;

    let mut c = DMatrix::repeat(
        network.voltage_source_count() + nodal_equations,
        edge_count,
        0.0,
    );

    // Add the current source coupling
    for (mut row_c, row_m) in c.row_iter_mut().zip(edge_to_node.row_iter()) {
        for edge in network.graph().edge_references() {
            if *edge.weight() == Type::Current {
                row_c[edge.id().index()] = row_m[edge.id().index()];
            }
        }
    }

    /*
    For each voltage source, use a row and add a "1" at the position of the corresponding edge
     */
    for (row_delta, col_idx) in network
        .graph()
        .edge_references()
        .filter_map(|e| {
            if *e.weight() == Type::Voltage {
                return Some(e.id().index());
            }
            return None;
        })
        .enumerate()
    {
        let row_idx = row_delta + nodal_equations;
        c[(row_idx, col_idx)] = 1.0;
    }

    return c;
}

/**
Coupling matrix "B" between nodes and voltages. As described here: https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA3.html

The B matrix is an n×m matrix with only 0, 1 and -1 elements. Each location in the matrix corresponds to a particular voltage source (first dimension) or a node (second dimension).
If the positive terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a 1.
If the negative terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a -1.
All other elements of the B matrix are 0
 */
fn coupling_voltage_source_to_system(
    network: &Network,
    edge_to_node: &DMatrix<f64>,
) -> DMatrix<f64> {
    // Preallocate coupling matrix
    let number_voltage_sources = network.voltage_source_count();
    let mut coupling_matrix = DMatrix::repeat(edge_to_node.nrows(), number_voltage_sources, 0.0);

    /*
    From https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA3.html:
    The B matrix is an n×m matrix with only 0, 1 and -1 elements. Each location in the matrix corresponds to a particular
    voltage source (first dimension) or a node (second dimension). If the positive terminal of the ith voltage source is
    connected to node k, then the element (i,k) in the B matrix is a 1. If the negative terminal of the ith voltage source
    is connected to node k, then the element (i,k) in the B matrix is a -1. Otherwise, elements of the B matrix are zero.
    */
    for (col, (edge_idx, _)) in network
        .graph()
        .edge_weights()
        .cloned()
        .enumerate()
        .filter(|(_, edge_type)| *edge_type == Type::Voltage)
        .enumerate()
    {
        for (row, row_vals) in edge_to_node.row_iter().enumerate() {
            coupling_matrix[(row, col)] = row_vals[edge_idx];
        }
    }

    return coupling_matrix;
}

/**
 The system matrix A is assembled from four different matrices G, B, C and D as described in
https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA3.html:
* The A matrix is (m+n)×(m+n) (n is the number of nodes, and m is the number of independent voltage sources). It is created as [G B; C D].
* The G matrix is n×n and is determined by the interconnections between the passive circuit elements (resistors)
* The B matrix is n×m and is determined by the connection of the voltage sources.
* The C matrix is m×n and is determined by the connection of the voltage sources. ( B and C are closely related, particularly when only independent sources are considered).
* The D matrix is m×m and is zero if only independent sources are considered.
*/
fn initialize_system_matrix(network: &Network, edge_to_node: &DMatrix<f64>) -> DMatrix<f64> {
    let mat_b = coupling_voltage_source_to_system(network, edge_to_node);
    let num_voltage_sources = mat_b.ncols();
    let num_node_eq = edge_to_node.nrows();
    let dim = num_voltage_sources + num_node_eq;
    let mut system_matrix = DMatrix::repeat(dim, dim, 0.0);

    // Add the B matrix (https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA3.html#B_matrix)
    for row in 0..mat_b.nrows() {
        for col in 0..mat_b.ncols() {
            system_matrix[(row, num_node_eq + col)] = mat_b[(row, col)];
        }
    }

    // Add the C matrix (transpose of B matrix)
    for row in 0..mat_b.nrows() {
        for col in 0..mat_b.ncols() {
            system_matrix[(num_node_eq + col, row)] = mat_b[(row, col)];
        }
    }

    return system_matrix;
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
    fn test_coupling_edge_to_node() {
        let network = network_creation(true, true);
        let (m, zero_potential_node) = coupling_edge_to_node(&network);
        assert_eq!(zero_potential_node, 2);
        assert_eq!(m.ncols(), network.graph().edge_count());
        assert_eq!(m.nrows(), network.graph().node_count() - 1);

        let r0: Vec<i32> = m.row(0).iter().map(|v| *v as i32).collect();
        let r1: Vec<i32> = m.row(1).iter().map(|v| *v as i32).collect();
        let r2: Vec<i32> = m.row(2).iter().map(|v| *v as i32).collect();
        let r3: Vec<i32> = m.row(3).iter().map(|v| *v as i32).collect();

        assert_eq!(r0, vec![1, -1, 0, 0, 0, 0, 0]);
        assert_eq!(r1, vec![-1, 0, 1, 1, 0, 0, 0]);
        assert_eq!(r2, vec![0, 0, 0, -1, 1, 0, -1]);
        assert_eq!(r3, vec![0, 0, 0, 0, 0, -1, 1]);
    }

    #[test]
    fn test_coupling_excitation() {
        {
            let network = network_creation(true, true);
            let (m, _) = coupling_edge_to_node(&network);
            let c = coupling_excitation_edge_to_node(&network, &m);

            // Two voltage sources: c has two more rows than nodal equations
            assert_eq!(c.ncols(), network.graph().edge_count());
            assert_eq!(c.nrows(), network.graph().node_count() + 1);

            // No current sources
            let r0: Vec<i32> = c.row(0).iter().map(|v| *v as i32).collect();
            let r1: Vec<i32> = c.row(1).iter().map(|v| *v as i32).collect();
            let r2: Vec<i32> = c.row(2).iter().map(|v| *v as i32).collect();
            let r3: Vec<i32> = c.row(3).iter().map(|v| *v as i32).collect();

            assert_eq!(r0, vec![0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(r1, vec![0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(r2, vec![0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(r3, vec![0, 0, 0, 0, 0, 0, 0]);

            // Voltage sources
            let r4: Vec<i32> = c.row(4).iter().map(|v| *v as i32).collect();
            let r5: Vec<i32> = c.row(5).iter().map(|v| *v as i32).collect();

            assert_eq!(r4, vec![1, 0, 0, 0, 0, 0, 0]);
            assert_eq!(r5, vec![0, 0, 0, 0, 0, 1, 0]);
        }
        {
            let network = network_creation(false, true);
            let (m, _) = coupling_edge_to_node(&network);
            let c = coupling_excitation_edge_to_node(&network, &m);

            // One voltage source: c has one more row than nodal equations
            assert_eq!(c.ncols(), network.graph().edge_count());
            assert_eq!(c.nrows(), network.graph().node_count());

            // One current source at 0
            let r0: Vec<i32> = c.row(0).iter().map(|v| *v as i32).collect();
            let r1: Vec<i32> = c.row(1).iter().map(|v| *v as i32).collect();
            let r2: Vec<i32> = c.row(2).iter().map(|v| *v as i32).collect();
            let r3: Vec<i32> = c.row(3).iter().map(|v| *v as i32).collect();

            assert_eq!(r0, vec![1, 0, 0, 0, 0, 0, 0]);
            assert_eq!(r1, vec![-1, 0, 0, 0, 0, 0, 0]);
            assert_eq!(r2, vec![0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(r3, vec![0, 0, 0, 0, 0, 0, 0]);

            // Voltage sources
            let r4: Vec<i32> = c.row(4).iter().map(|v| *v as i32).collect();

            assert_eq!(r4, vec![0, 0, 0, 0, 0, 1, 0]);
        }
        {
            let network = network_creation(true, false);
            let (m, _) = coupling_edge_to_node(&network);
            let c = coupling_excitation_edge_to_node(&network, &m);

            // One voltage source: c has one more row than nodal equations
            assert_eq!(c.ncols(), network.graph().edge_count());
            assert_eq!(c.nrows(), network.graph().node_count());

            // One current source at 5
            let r0: Vec<i32> = c.row(0).iter().map(|v| *v as i32).collect();
            let r1: Vec<i32> = c.row(1).iter().map(|v| *v as i32).collect();
            let r2: Vec<i32> = c.row(2).iter().map(|v| *v as i32).collect();
            let r3: Vec<i32> = c.row(3).iter().map(|v| *v as i32).collect();

            assert_eq!(r0, vec![0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(r1, vec![0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(r2, vec![0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(r3, vec![0, 0, 0, 0, 0, -1, 0]);

            // Voltage sources
            let r4: Vec<i32> = c.row(4).iter().map(|v| *v as i32).collect();

            assert_eq!(r4, vec![1, 0, 0, 0, 0, 0, 0]);
        }
        {
            let network = network_creation(false, false);
            let (m, _) = coupling_edge_to_node(&network);
            let c = coupling_excitation_edge_to_node(&network, &m);

            // No voltage source: The row number of c is equal to the number of nodal equations
            assert_eq!(c.ncols(), network.graph().edge_count());
            assert_eq!(c.nrows(), network.graph().node_count() - 1);

            // Two current sources
            let r0: Vec<i32> = c.row(0).iter().map(|v| *v as i32).collect();
            let r1: Vec<i32> = c.row(1).iter().map(|v| *v as i32).collect();
            let r2: Vec<i32> = c.row(2).iter().map(|v| *v as i32).collect();
            let r3: Vec<i32> = c.row(3).iter().map(|v| *v as i32).collect();

            assert_eq!(r0, vec![1, 0, 0, 0, 0, 0, 0]);
            assert_eq!(r1, vec![-1, 0, 0, 0, 0, 0, 0]);
            assert_eq!(r2, vec![0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(r3, vec![0, 0, 0, 0, 0, -1, 0]);
        }
    }

    #[test]
    fn test_coupling_edge_to_node_resistance() {
        // Same for all source types
        let network = network_creation(false, false);
        let (m, _) = coupling_edge_to_node(&network);
        let r = coupling_edge_to_node_resistance(&network, &m);

        // Diagonals
        assert_eq!(r[(0, 0)], vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(r[(1, 1)], vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(r[(2, 2)], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        assert_eq!(r[(3, 3)], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);

        // Off-diagonals
        assert_eq!(r[(0, 1)], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(r[(0, 1)], r[(1, 0)]);

        assert_eq!(r[(0, 2)], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(r[(0, 2)], r[(2, 0)]);

        assert_eq!(r[(0, 3)], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(r[(0, 3)], r[(3, 0)]);

        assert_eq!(r[(1, 2)], vec![0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]);
        assert_eq!(r[(1, 2)], r[(2, 1)]);

        assert_eq!(r[(1, 3)], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(r[(1, 3)], r[(3, 1)]);

        assert_eq!(r[(2, 3)], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]);
        assert_eq!(r[(2, 3)], r[(3, 2)]);
    }

    #[test]
    fn test_coupling_voltage_source_to_system() {
        {
            let network = network_creation(false, false);
            // Output matrix has dimension 4x0 (no voltage sources)
            let (m, _) = coupling_edge_to_node(&network);
            let c = coupling_voltage_source_to_system(&network, &m);
            assert_eq!(c.ncols(), network.voltage_source_count());
            assert_eq!(c.nrows(), network.graph().node_count() - 1);
        }
        {
            let network = network_creation(true, false);
            // Output matrix has dimension 4x1 (one voltage source)
            let (m, _) = coupling_edge_to_node(&network);
            let c = coupling_voltage_source_to_system(&network, &m);
            assert_eq!(c.ncols(), network.voltage_source_count());
            assert_eq!(c.nrows(), network.graph().node_count() - 1);
            assert_eq!(c[(0, 0)], 1.0);
            assert_eq!(c[(1, 0)], -1.0);
            assert_eq!(c[(2, 0)], 0.0);
            assert_eq!(c[(3, 0)], 0.0);
        }
        {
            let network = network_creation(false, true);
            // Output matrix has dimension 4x1 (one voltage source)
            let (m, _) = coupling_edge_to_node(&network);
            let c = coupling_voltage_source_to_system(&network, &m);
            assert_eq!(c.ncols(), network.voltage_source_count());
            assert_eq!(c.nrows(), network.graph().node_count() - 1);
            assert_eq!(c[(0, 0)], 0.0);
            assert_eq!(c[(1, 0)], 0.0);
            assert_eq!(c[(2, 0)], 0.0);
            assert_eq!(c[(3, 0)], -1.0);
        }
        {
            let network = network_creation(true, true);
            // Output matrix has dimension 4x1 (one voltage source)
            let (m, _) = coupling_edge_to_node(&network);
            let c = coupling_voltage_source_to_system(&network, &m);
            assert_eq!(c.ncols(), network.voltage_source_count());
            assert_eq!(c.nrows(), network.graph().node_count() - 1);
            assert_eq!(c[(0, 0)], 1.0);
            assert_eq!(c[(1, 0)], -1.0);
            assert_eq!(c[(2, 0)], 0.0);
            assert_eq!(c[(3, 0)], 0.0);

            assert_eq!(c[(0, 1)], 0.0);
            assert_eq!(c[(1, 1)], 0.0);
            assert_eq!(c[(2, 1)], 0.0);
            assert_eq!(c[(3, 1)], -1.0);
        }
    }

    #[test]
    fn test_unknowns_to_edge_voltage() {
        {
            // No voltage sources => the coupling matrix is simply the transposed version of edge_to_node
            let network = network_creation(false, false);
            let (m, _) = coupling_edge_to_node(&network);
            let c = -coupling_unknowns_to_edge_voltage(&network, &m);
            assert_eq!(m.transpose(), c);
        }
        {
            // One voltage source, one current source
            let network = network_creation(true, false);
            let (m, _) = coupling_edge_to_node(&network);
            let c = coupling_unknowns_to_edge_voltage(&network, &m);

            assert_eq!(m.ncols(), c.nrows());
            assert_eq!(m.nrows() + 1, c.ncols());
            for row in 0..m.nrows() {
                for col in 0..m.ncols() {
                    assert_eq!(m[(row, col)], -c[(col, row)]);
                }
            }

            // All values in the additional column are zero
            for val in c.column(m.nrows()).iter().cloned() {
                assert_eq!(val, 0.0);
            }
        }
        {
            // Two voltage sources
            let network = network_creation(true, true);
            let (m, _) = coupling_edge_to_node(&network);
            let c = coupling_unknowns_to_edge_voltage(&network, &m);

            assert_eq!(m.ncols(), c.nrows());
            assert_eq!(m.nrows() + 2, c.ncols());
            for row in 0..m.nrows() {
                for col in 0..m.ncols() {
                    assert_eq!(m[(row, col)], -c[(col, row)]);
                }
            }

            // All values in the additional columns are zero
            for val in c
                .column(m.nrows())
                .iter()
                .cloned()
                .chain(c.column(m.nrows() + 1).iter().cloned())
            {
                assert_eq!(val, 0.0);
            }
        }
    }

    #[test]
    fn test_system_matrix_init() {
        {
            let network = network_creation(false, false);
            let (m, _) = coupling_edge_to_node(&network);
            let sys_mat = initialize_system_matrix(&network, &m);
            for elem in sys_mat.iter().cloned() {
                assert_eq!(elem, 0.0);
            }
        }
        {
            let network = network_creation(true, false);
            let (m, _) = coupling_edge_to_node(&network);
            let sys_mat = initialize_system_matrix(&network, &m);
            for row in 0..sys_mat.nrows() {
                for col in 0..sys_mat.ncols() {
                    if row == 0 && col == 4 {
                        assert_eq!(sys_mat[(row, col)], 1.0);
                    } else if row == 1 && col == 4 {
                        assert_eq!(sys_mat[(row, col)], -1.0);
                    } else if col == 0 && row == 4 {
                        assert_eq!(sys_mat[(row, col)], 1.0);
                    } else if col == 1 && row == 4 {
                        assert_eq!(sys_mat[(row, col)], -1.0);
                    } else {
                        assert_eq!(sys_mat[(row, col)], 0.0);
                    }
                }
            }
        }
        {
            let network = network_creation(false, true);
            let (m, _) = coupling_edge_to_node(&network);
            let sys_mat = initialize_system_matrix(&network, &m);
            for row in 0..sys_mat.nrows() {
                for col in 0..sys_mat.ncols() {
                    if row == 3 && col == 4 {
                        assert_eq!(sys_mat[(row, col)], -1.0);
                    } else if col == 3 && row == 4 {
                        assert_eq!(sys_mat[(row, col)], -1.0);
                    } else {
                        assert_eq!(sys_mat[(row, col)], 0.0);
                    }
                }
            }
        }
        {
            let network = network_creation(true, true);
            let (m, _) = coupling_edge_to_node(&network);
            let sys_mat = initialize_system_matrix(&network, &m);
            for row in 0..sys_mat.nrows() {
                for col in 0..sys_mat.ncols() {
                    if row == 0 && col == 4 {
                        assert_eq!(sys_mat[(row, col)], 1.0);
                    } else if row == 1 && col == 4 {
                        assert_eq!(sys_mat[(row, col)], -1.0);
                    } else if col == 0 && row == 4 {
                        assert_eq!(sys_mat[(row, col)], 1.0);
                    } else if col == 1 && row == 4 {
                        assert_eq!(sys_mat[(row, col)], -1.0);
                    } else if row == 3 && col == 5 {
                        assert_eq!(sys_mat[(row, col)], -1.0);
                    } else if col == 3 && row == 5 {
                        assert_eq!(sys_mat[(row, col)], -1.0);
                    } else {
                        assert_eq!(sys_mat[(row, col)], 0.0);
                    }
                }
            }
        }
    }
}
