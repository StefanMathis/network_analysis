/*!
Tools for creating a [`Network`].

This module provides the [`Network`] struct, which is used to construct [`MeshAnalysis`](crate::MeshAnalysis) and
[`NodalAnalysis`](crate::NodalAnalysis) instances. A [`Network`] is essentially a wrapper around a
[`UnGraph<usize, Excitation>`] which makes sure the network represented by the graph is suitable
for these network analysis methods. Furthermore, this module also contains the [`BuildError`] struct,
which covers the different reasons why a graph might not be a valid [`Network`]. See their respective
docstrings for further information.
*/

use std::{cmp::PartialEq, collections::VecDeque};

use petgraph::{
    graph::{EdgeIndex, NodeIndex, UnGraph},
    stable_graph::StableUnGraph,
    visit::{EdgeRef, IntoEdgeReferences},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/**
An error returned from a failed attempt of creating a [`Network`] from a graph.

The diagram below shows a graph which fails to represent a valid electrical circuit:
```text
 ┌──┬─[1]─┬─[5]─
 │  │    [2]
 │ [0]    │
 │  │    [3]
 └──┴─[4]─┘
```
- Edge 0 is short-circuited. To fix this issue, it needs to be removed from the network.
- If both edge 2 and 4 would have a current excitation, the current flowing through the edges 2, 3 and 4
cannot be defined. To fix this issue, one of the excitations needs to be removed.
- Edge 5 is a dead end. It can simply be deleted.
 */
#[derive(Debug, Clone)]
pub enum BuildError {
    /**
    Two (or more) current excitations are defined without a "junction node" (more than two edges) in between.
    For example, if a node is used by only two edges and both edges have a current excitation, the current going
    through the edges is not unambiguous (except for the trivial case of both current excitations being equal,
    however in this case one of them can be omitted anyway).

    Variant contains the indices of the two offending current sources and the input graph.
     */
    TwoCurrentExcitations {
        first_edge: EdgeIndex,
        second_edge: EdgeIndex,
        graph: UnGraph<usize, Type>,
    },
    /**
    A node is a "dead end" (has only one edge using it). Since the circuit is not closed, no current can go through
    this edge and it can be omitted.

    Variant contains the index of the "dead end" node and the input graph.
     */
    DeadEnd {
        node: NodeIndex,
        graph: UnGraph<usize, Type>,
    },
    /**
    An edge is short-circuited (source and target node are identical). In this case, the edge can be omitted.

    Variant contains the index of the short-circuited edge and the input graph.
     */
    ShortCircuit {
        edge: EdgeIndex,
        graph: UnGraph<usize, Type>,
    },
    /**
    Some edges are not connected to other edges, or in other words, the graph represents two or more circuits instead of one.
     */
    NotAllEdgesConnected,
    /**
    For each edge, at least one of the possible loops containing it must not have more than one current source.
    Otherwise, the network cannot be solved for the general case, since some current sources are contradicting.

    Example: If 0, 6 and 3 are current sources, the network is overdefined, since there is no possible loop without two current sources.
    ```text
     ┌─[1]─┬─[2]─┐
    [0]   [6]   [3]
     └─[5]─┴─[4]─┘
    ```
    Variant contains the index of one current source edge which must be changed to make the graph valid as well as the input graph.
     */
    OverdefinedByCurrentSources {
        edge: EdgeIndex,
        graph: UnGraph<usize, Type>,
    },
    /**
    For each edge, all possible loops containing it must not have only voltage sources.
    Otherwise, the network cannot be solved for the general case, since some voltage sources are contradicting.

    Example: If 0 and 1 are voltage sources, the network is overdefined, since the loop 0 -> 1 contains only voltage sources.
    ```text
     ┌───┬───┐
    [0] [1] [2]
     └───┴───┘
    ```
    Variant contains the index of one voltage source edge which must be changed to make the graph valid as well as the input graph.
     */
    OverdefinedByVoltageSources {
        edge: EdgeIndex,
        graph: UnGraph<usize, Type>,
    },
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::TwoCurrentExcitations {
                first_edge,
                second_edge,
                graph: _,
            } => write!(
                f,
                "the edges {} and {} both have a current source and are connected without a fork
                (node with at least three edges) inbetween. The resulting network is not solvable.",
                first_edge.index(),
                second_edge.index()
            ),
            BuildError::DeadEnd { node, graph: _ } => write!(
                f,
                "node {} is a dead end (has only one edge using it). The resulting network is not solvable.",
                node.index()
            ),
            BuildError::ShortCircuit { edge, graph } => write!(
                f,
                "edge {} has two identical end points (node {}). The resulting network is not solvable.",
                edge.index(),
                graph.edge_endpoints(*edge).expect("edge exists").0.index()
            ),
            BuildError::NotAllEdgesConnected => write!(f, "not all edges are connected"),
            BuildError::OverdefinedByCurrentSources { edge, graph: _ } => {
                write!(
                    f,
                    "network is overdefined by current sources. Remove the source in edge {}",
                    edge.index()
                )
            }
            BuildError::OverdefinedByVoltageSources { edge, graph: _ } => {
                write!(
                    f,
                    "network is overdefined by voltage sources. Remove the source in edge {}",
                    edge.index()
                )
            }
        }
    }
}

impl std::error::Error for BuildError {}

/**
An enum representing the different physical quantities within the [`Network`].

These types are not necessarily electrical quantities, but can represent any similar
physical quantities which follow the physical relationship `voltage = current * resistance`.

Examples:
- Magnetic domain: Magnetic voltage (voltage), magnetic flux (current), reluctance (resistance)
- Thermal domain: Temperature difference (voltage), power (current), thermal resistance (resistance)

# Features

This enum can be serialized / deserialized via the [serde](https://crates.io/crates/serde)
crate if the `serde` feature is enabled.
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Type {
    Voltage,
    Current,
    Resistance,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let t = match self {
            Type::Voltage => "voltage",
            Type::Current => "current",
            Type::Resistance => "resistance",
        };
        write!(f, "{t}")
    }
}

/**
An edge where the nodes are defined implicitly via the other edges using them.

The `edge_type` describes whether the edge is a resistance, a current or a voltage source.
If the edge has an excitation, it is always oriented from source to target.
The index of each `EdgeListEdge` in `network` equals the index of the represented edge itself.

# Examples
```
/*
Network structure:
 ┌──[0]──┐
 │       │
[3]     [1]
 │       │
 └──[2]──┘
Voltage source oriented from 1 to 3 at edge 0
*/
use network_analysis::{EdgeListEdge, Network, Type};
assert!(Network::from_edge_list_edges(
    &[
            EdgeListEdge::new(vec![3], vec![1], Type::Voltage),
            EdgeListEdge::new(vec![0], vec![2], Type::Resistance),
            EdgeListEdge::new(vec![1], vec![3], Type::Resistance),
            EdgeListEdge::new(vec![2], vec![0], Type::Resistance),
        ]
    ).is_ok());

/*
Network structure:
 ┌──[0]──┬──[1]──┐
 │       │       │
[2]     [3]     [4]
 │       │       │
 └──[5]──┴──[6]──┘
Current source oriented from 5 to 0 at edge 2
*/
assert!(Network::from_edge_list_edges(
    &[
            EdgeListEdge::new(vec![3, 1], vec![2], Type::Resistance),
            EdgeListEdge::new(vec![0, 3], vec![4], Type::Resistance),
            EdgeListEdge::new(vec![5], vec![0], Type::Current),
            EdgeListEdge::new(vec![0, 1], vec![5, 6], Type::Resistance),
            EdgeListEdge::new(vec![1], vec![6], Type::Resistance),
            EdgeListEdge::new(vec![2], vec![3, 6], Type::Resistance),
            EdgeListEdge::new(vec![3, 5], vec![4], Type::Resistance),
        ]
    ).is_ok());
```

# Features

This struct can be serialized / deserialized via the [serde](https://crates.io/crates/serde)
crate if the `serde` feature is enabled.
 */
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EdgeListEdge {
    pub source: Vec<usize>,
    pub target: Vec<usize>,
    pub edge_type: Type,
}

impl EdgeListEdge {
    /**
    Creates a new instance of `Self` from its components.
     */
    pub fn new(source: Vec<usize>, target: Vec<usize>, edge_type: Type) -> Self {
        return Self {
            source,
            target,
            edge_type,
        };
    }
}

/**
An edge where the node indices are specified explictly.

The `edge_type` describes whether the edge is a resistance, a current or a voltage source.
If the edge has an excitation, it is always oriented from source to target.

# Examples

```
use network_analysis::{NodeEdge, Network, Type};

/*
Valid input: The node indices start at 0 and form a continuous number sequence up to 3

(0)──[0]──(2)
 │         │
[3]       [1]
 │         │
(3)──[2]──(1)
    */
assert!(Network::from_node_edges(
    &[
            NodeEdge::new(0, 2, Type::Voltage),
            NodeEdge::new(2, 1, Type::Resistance),
            NodeEdge::new(1, 3, Type::Resistance),
            NodeEdge::new(3, 0, Type::Resistance),
        ]
    ).is_ok());


/*
Invalid input: The node indices do not start at zero and do not form a continuous sequence.

(1)──[0]──(3)
 │         │
[3]       [1]
 │         │
(6)──[2]──(7)
    */

assert!(Network::from_node_edges(
    &[
        NodeEdge::new(1, 3, Type::Voltage),
        NodeEdge::new(3, 7, Type::Resistance),
        NodeEdge::new(7, 6, Type::Resistance),
        NodeEdge::new(6, 1, Type::Resistance),
    ]
).is_err());
```

# Features

This struct can be serialized / deserialized via the [serde](https://crates.io/crates/serde)
crate if the `serde` feature is enabled.
 */
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NodeEdge {
    pub source: usize,
    pub target: usize,
    pub edge_type: Type,
}

impl NodeEdge {
    /**
    Creates a new instance of `Self` from its components.
     */
    pub fn new(source: usize, target: usize, edge_type: Type) -> Self {
        return Self {
            source,
            target,
            edge_type,
        };
    }
}

/**
A network which is valid for performing mesh and nodal analysis.

This struct represents a network which can be solved via [`MeshAnalysis`](crate::mesh_analysis::MeshAnalysis) and
[`NodalAnalysis`](crate::nodal_analysis::NodalAnalysis). It is essentially a
[`newtype`](https://doc.rust-lang.org/rust-by-example/generics/new_types.html) wrapper around a
[`UnGraph<usize, Excitation>`] which forms a valid electrical circuit. Please see [`BuildError`] for
all the possible ways a graph can fail to represent a valid electrical circuit.

# Sign conventions

The sign of an edge is oriented from source to target. If the following network has a current source at edge 0
and resistances everywhere else and all edges are oriented clockwise (e.g source of edge 0 is at edge 3 and
target of edge 0 is at edge 1), an input of -5 at the current source results in an edge current of -5 over all edges.
```text
     ┌──[1]──┐
  |  │       │
 5| [0]     [2]
  V  │       │
     └──[3]──┘
```

Likewise, if edge 2 is "flipped" (source at edge 3, target at edge 1), its edge current would be +5.

If edge 0 is a voltage source with value +3 instead (i.e high potential at the source, low potential at the target)
and all resistances have the value 1 (resistances must always be positive),
the current going through all edges would be -1 (since the voltage drop must be -1 over all resistances so the entire
loop adds up to a voltage of 0).
```text
     ┌──[1]──┐
  ^  │       │
 3| [0]     [2]
  |  │       │
     └──[3]──┘
```

# Constructing a [`Network`]

The following constructor methods are available:
- [`Network::new`] (using a graph)
- [`Network::from_node_edges`] (using a slice of [`NodeEdge`])
- [`Network::from_edge_list_edges`] (using a slice of [`EdgeListEdge`])

Please see the docstrings of the methods for examples.

# Serialization and deserialization

Serialization and deserialization requires that the `serde` feature is enabled.
This struct serializes into a `Vec<NodeEdge>` and can be (fallible) deserialized from the following types:
- `Vec<NodeEdge>`
- `Vec<EdgeListEdge>`
- `UnGraph<usize, Type>`
 */
#[repr(transparent)]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Network(UnGraph<usize, Type>);

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Network {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let content =
            <serde::__private::de::Content as serde::Deserialize>::deserialize(deserializer)?;
        let deserializer = serde::__private::de::ContentRefDeserializer::<D::Error>::new(&content);

        /*
        Try to deserialize this struct from either a
        - `UnGraph<usize, Type>`
        - `Vec<NodeEdge>`
        - `Vec<EdgeListEdge>`
         */
        if let Ok(graph) = UnGraph::<usize, Type>::deserialize(deserializer) {
            return Self::new(graph).map_err(serde::de::Error::custom);
        }
        if let Ok(coll) = Vec::<NodeEdge>::deserialize(deserializer) {
            return Self::from_node_edges(coll.as_slice()).map_err(serde::de::Error::custom);
        }
        if let Ok(coll) = Vec::<EdgeListEdge>::deserialize(deserializer) {
            return Self::from_edge_list_edges(coll.as_slice()).map_err(serde::de::Error::custom);
        }

        return Err(serde::de::Error::custom(
            "can only deserialize from the following types: Vec<NodeEdge>, Vec<EdgeListEdge>",
        ));
    }
}

impl Network {
    /**
    A graph is a valid network if it fulfills the conditions outlined in the docstring of [`Network`].
    The edge weights define whether the edge is a resistance, a current source or a voltage source.
     */
    pub fn new(graph: UnGraph<usize, Type>) -> Result<Self, BuildError> {
        // Condition 1)
        if petgraph::algo::connected_components(&graph) != 1 {
            return Err(BuildError::NotAllEdgesConnected);
        }

        // Condition 2)
        for edge in graph.edge_references() {
            if edge.source() == edge.target() {
                return Err(BuildError::ShortCircuit {
                    edge: edge.id(),
                    graph,
                });
            }
        }

        // Condition 3)
        for node_idx in graph.node_indices() {
            if graph.edges(node_idx).count() < 2 {
                return Err(BuildError::DeadEnd {
                    node: node_idx,
                    graph,
                });
            }
        }

        // Needed for condition 4) and 5)
        let spanning_tree = find_spanning_tree(&graph);

        // Condition 4)
        // It is sufficient to check if each independent branch has at most one current source,
        // since the network fails condition 5) anyway if two current sources are on the same branch
        // in the spanning tree
        for branch in spanning_tree.independent_branches.iter() {
            let mut first_current_source: Option<EdgeIndex> = None;
            for edge in branch.edges.iter().cloned() {
                if let Some(edge_type) = graph.edge_weight(edge) {
                    if *edge_type == Type::Current {
                        /*
                        If the branch already contains another current source, condition 4 is failed
                         */
                        match first_current_source {
                            Some(first_edge) => {
                                return Err(BuildError::TwoCurrentExcitations {
                                    first_edge,
                                    second_edge: edge,
                                    graph,
                                });
                            }
                            None => {
                                first_current_source = Some(edge);
                            }
                        }
                    }
                }
            }
        }

        // Condition 5)
        for edge_ref in spanning_tree.graph.edge_references() {
            if *edge_ref.weight() == Type::Current {
                return Err(BuildError::OverdefinedByCurrentSources {
                    edge: edge_ref.id(),
                    graph,
                });
            }
        }

        // Condition 6)
        for independent_branch in spanning_tree.independent_branches.iter() {
            /*
            If the independent branch contains only voltages, check if the corresponding part
            of the spanning tree also contains only voltage sources.
             */
            if independent_branch
                .edges
                .iter()
                .all(|idx| *graph.edge_weight(*idx).expect("must exist") == Type::Voltage)
            {
                /*
                To see if the connection between source and target of the spanning tree contains only voltage sources,
                we use Dijkstra's algorithm and assign any edges which are voltage sources the value 0 and all
                other edges the value 1. If the total cost calculated by Dijkstra's algorithm is zero,
                the network is overdefined by voltage sources.
                 */
                let val = petgraph::algo::bidirectional_dijkstra(
                    &spanning_tree.graph,
                    independent_branch.source,
                    independent_branch.target,
                    |edge| (*edge.weight() != Type::Voltage) as u32,
                );
                if val == Some(0) {
                    // We can return any of the voltage sources on the independent branch here
                    return Err(BuildError::OverdefinedByVoltageSources {
                        edge: independent_branch.edges[0],
                        graph,
                    });
                }
            }
        }

        return Ok(Self(graph));
    }

    /**
    Creates a new instance of `Self` from a slice of [`NodeEdge`].
    See the docstring of [`NodeEdge`] for an example.
     */
    pub fn from_node_edges(edges: &[NodeEdge]) -> Result<Self, BuildError> {
        // Find the largest node index specified in one of the `NodeEdge`s
        let mut max_node = 0;
        for edge in edges.iter() {
            if edge.source > max_node {
                max_node = edge.source;
            }
            if edge.target > max_node {
                max_node = edge.target;
            }
        }

        // Assumption: Number of nodes approximately equal to the number of edges
        let mut graph = UnGraph::<usize, Type>::with_capacity(edges.len(), edges.len());
        for node_tag in 0..(max_node + 1) {
            graph.add_node(node_tag);
        }
        for edge in edges.iter() {
            let source = NodeIndex::new(edge.source);
            let target = NodeIndex::new(edge.target);
            graph.add_edge(source, target, edge.edge_type); // Note that this method allows parallel edges!
        }
        return Self::new(graph);
    }

    /**
    Creates a new instance of `Self` from a slice of [`EdgeListEdge`].

    See the docstring of [`EdgeListEdge`] for an example.
     */
    pub fn from_edge_list_edges(edge_list_edges: &[EdgeListEdge]) -> Result<Self, BuildError> {
        // Intialize an empty vector to hold the edge definition
        let num_edges = edge_list_edges.len();
        let mut node_edges: Vec<NodeEdge> = vec![
            NodeEdge {
                source: 0,
                target: 0,
                edge_type: Type::Resistance,
            };
            num_edges
        ];

        // Node list describing the edges connected to this node.
        let mut nodes: Vec<Vec<usize>> = Vec::new();

        // Node tag counter
        let mut free_node_tag: usize = 0;

        // Populate edges
        for (edge_tag, (mut edge_by_node, edge_by_edge)) in node_edges
            .iter_mut()
            .zip(edge_list_edges.iter())
            .enumerate()
        {
            // Inherit the source properties from `edge_by_edge`
            edge_by_node.edge_type = edge_by_edge.edge_type;

            for (i_side, side) in [&edge_by_edge.source, &edge_by_edge.target]
                .iter()
                .enumerate()
            {
                // Create a temporary node from the 'source' or 'target' side of the edge
                let mut edges_connected_to_this_node = side.to_vec(); // All edges listed as connected to the current edge
                edges_connected_to_this_node.push(edge_tag.try_into().unwrap()); // add the current edge itself, which has the tag "i"

                // Sort the edge vector, this is necessary for later comparison
                edges_connected_to_this_node.sort_unstable(); // => https://rust-lang-nursery.github.io/rust-cookbook/algorithms/sorting.html

                // Check if the temporary node is already included in edges_connected_to_this_node.
                // If not, add it to that list. Add the node index of the current side
                // to the path dictionary.
                let mut index = None;
                for (node_idx, node) in nodes.iter().enumerate() {
                    // If we're searching the target node, skip the source node
                    if i_side == 1 && edge_by_node.source == node_idx {
                        continue;
                    }
                    if slices_are_identical(node.as_slice(), &edges_connected_to_this_node) {
                        index = Some(node_idx);
                        break;
                    }
                }

                match index {
                    // The node has not been created yet => create it now!
                    None => {
                        update_edge_by_node(&mut edge_by_node, free_node_tag, i_side);
                        nodes.push(edges_connected_to_this_node);

                        // Now update the free node tag
                        free_node_tag += 1;
                    }
                    // The node does already exist => update the edge with this node
                    Some(existing_node_tag) => {
                        update_edge_by_node(&mut edge_by_node, existing_node_tag, i_side);
                    }
                }
            }
        }

        return Self::from_node_edges(node_edges.as_slice());
    }

    /**
    Accesses the [`petgraph::stable_graph::UnGraph`] representation of the network.

    The network analysis structs [`MeshAnalysis`](crate::mesh_analysis::MeshAnalysis) and
    [`NodalAnalysis`](crate::nodal_analysis::NodalAnalysis) use [`petgraph`](https://crates.io/crates/petgraph)s
    [`UnGraph`] when they are created from a [`Network`] instance. This method allows accessing
    the graph directly. Please be aware that [`Network`] holds more information than just the graph
    (i.e. the edge source type information) and can therefore not be losslessly represented by a [`UnGraph`]
    (otherwise, the need for this type would not exist in the first place).

    # Examples
    ```
    use network_analysis::{EdgeListEdge, Network, Type};
    use petgraph::algo::dijkstra;
    use petgraph::visit::NodeIndexable;

    let network = Network::from_edge_list_edges(
    &[
            EdgeListEdge::new(vec![3], vec![1], Type::Voltage),
            EdgeListEdge::new(vec![0], vec![2], Type::Resistance),
            EdgeListEdge::new(vec![1], vec![3], Type::Resistance),
            EdgeListEdge::new(vec![2], vec![0], Type::Resistance),
        ]
    ).expect("valid network");
    let g = network.graph();

    // Now use some of the functionality provided by petgraph, e.g. the "dijkstra" path finding algorithm
    let node_map = dijkstra(&g, 0.into(), Some(2.into()), |_| 1);
    assert_eq!(&2i32, node_map.get(&g.from_index(2)).unwrap());
    ```
     */
    pub fn graph(&self) -> &UnGraph<usize, Type> {
        return &self.0;
    }

    /**
    Returns the number of voltage sources by counting all edges where `edge.edge_type == Type::Voltage`.

    # Examples
    ```
    use network_analysis::{EdgeListEdge, Network, Type};

    let network = Network::from_edge_list_edges(
    &[
            EdgeListEdge::new(vec![3], vec![1], Type::Voltage),
            EdgeListEdge::new(vec![0], vec![2], Type::Resistance),
            EdgeListEdge::new(vec![1], vec![3], Type::Current),
            EdgeListEdge::new(vec![2], vec![0], Type::Voltage),
        ]
    ).expect("valid network");
    assert_eq!(network.voltage_source_count(), 2);
    ```
     */
    pub fn voltage_source_count(&self) -> usize {
        return self
            .0
            .edge_weights()
            .map(|edge_type| (edge_type == &Type::Voltage) as usize)
            .sum();
    }

    /**
    Returns the number of current sources by counting all edges where `edge.edge_type == Type::Current`.

    # Examples
    ```
    use network_analysis::{EdgeListEdge, Network, Type};

    let network = Network::from_edge_list_edges(
    &[
            EdgeListEdge::new(vec![3], vec![1], Type::Voltage),
            EdgeListEdge::new(vec![0], vec![2], Type::Resistance),
            EdgeListEdge::new(vec![1], vec![3], Type::Current),
            EdgeListEdge::new(vec![2], vec![0], Type::Voltage),
        ]
    ).expect("valid network");
    assert_eq!(network.current_source_count(), 1);
    ```
     */
    pub fn current_source_count(&self) -> usize {
        return self
            .0
            .edge_weights()
            .map(|edge_type| (edge_type == &Type::Current) as usize)
            .sum();
    }
}

pub(crate) struct IndependentBranch {
    pub(crate) edges: VecDeque<EdgeIndex>,
    pub(crate) coupling: VecDeque<bool>,
    pub(crate) source: NodeIndex,
    pub(crate) target: NodeIndex,
}

pub(crate) struct SpanningTree {
    pub(crate) graph: StableUnGraph<usize, Type>,
    pub(crate) independent_branches: Vec<IndependentBranch>,
    pub(crate) number_current_source_meshes: usize,
}

pub(crate) fn find_spanning_tree(graph: &UnGraph<usize, Type>) -> SpanningTree {
    fn try_remove_adjacent_edge(
        graph: &UnGraph<usize, Type>,
        spanning_tree: &mut StableUnGraph<usize, Type>,
        independent_branch: &mut IndependentBranch,
        start_from_source: bool,
    ) {
        let previous_edge_index = if start_from_source {
            independent_branch
                .edges
                .back()
                .expect("This function is only called when independent_branch.edges already has an element populated.")
        } else {
            independent_branch
                .edges
                .front()
                .expect("This function is only called when independent_branch.edges already has an element populated.")
        };

        let branch_end_node = if start_from_source {
            &mut independent_branch.source
        } else {
            &mut independent_branch.target
        };

        // Check how many edges are connected to the given node. If only one is connected, remove it and repeat the process
        let mut single_edge: Option<EdgeIndex> = None;
        for (idx, neighbor) in graph.edges(*branch_end_node).enumerate() {
            if idx < 2 && &neighbor.id() != previous_edge_index {
                single_edge = Some(neighbor.id());
            } else if idx > 1 {
                single_edge = None;
                break;
            }
        }
        match single_edge {
            Some(edge_index) => {
                // If this function returns None, the entire network is actually a single mesh
                match spanning_tree.edge_endpoints(edge_index) {
                    Some((a, b)) => {
                        spanning_tree.remove_edge(edge_index);

                        // Identify the coupling between previous_edge_index and edge_index
                        let coupling = {
                            let (source, target) =
                                unsafe { graph.edge_endpoints(edge_index).unwrap_unchecked() };
                            let (prev_source, prev_target) = unsafe {
                                graph
                                    .edge_endpoints(*previous_edge_index)
                                    .unwrap_unchecked()
                            };
                            if start_from_source {
                                prev_source == target // If true, the edges are oriented in the same direction.
                            } else {
                                prev_target == source // If true, the edges are oriented in the same direction.
                            }
                        };

                        // Add the edge index and the coupling
                        if start_from_source {
                            independent_branch.edges.push_front(edge_index);
                            independent_branch.coupling.push_front(coupling);
                        } else {
                            independent_branch.edges.push_back(edge_index);
                            independent_branch.coupling.push_back(coupling);
                        }

                        if a == *branch_end_node {
                            *branch_end_node = b;
                            return try_remove_adjacent_edge(
                                graph,
                                spanning_tree,
                                independent_branch,
                                start_from_source,
                            );
                        } else {
                            *branch_end_node = a;
                            return try_remove_adjacent_edge(
                                graph,
                                spanning_tree,
                                independent_branch,
                                start_from_source,
                            );
                        }
                    }
                    None => return (),
                }
            }
            None => return (),
        }
    }

    fn try_create_new_mesh(
        graph: &UnGraph<usize, Type>,
        mut spanning_tree: &mut StableUnGraph<usize, Type>,
        independent_branches: &mut Vec<IndependentBranch>,
        edge_index: EdgeIndex,
    ) {
        // Find the nodes associated with the edge
        let (a, b) = graph.edge_endpoints(edge_index).unwrap();

        // Remove the "edge_index" edge from the spanning tree and check if the graph is still connected.
        if let Some(edge_type) = spanning_tree.remove_edge(edge_index) {
            // If true, the nodes a and b are still connected, hence the edge is not part of the spanning tree.
            // If false, the edge is part of the spanning tree and therefore it is added again to the spanning tree.
            if petgraph::algo::has_path_connecting(&*spanning_tree, a, b, None) {
                let mut branch = IndependentBranch {
                    edges: VecDeque::new(),
                    coupling: VecDeque::new(),
                    source: a,
                    target: b,
                };
                branch.edges.push_back(edge_index);
                branch.coupling.push_back(true);

                // If only one edge is attached to node a or b, this edge belongs to the independent branch as well.
                // Repeat this process until a node is encountered which features multiple edges connrected to it.
                try_remove_adjacent_edge(graph, &mut spanning_tree, &mut branch, true);
                try_remove_adjacent_edge(graph, &mut spanning_tree, &mut branch, false);

                // Add the edges to the list of independent branches
                independent_branches.push(branch);
            } else {
                // Readd the edge to the spanning tree
                spanning_tree.add_edge(a, b, edge_type);
            }
        }
    }

    /*
    Create a spanning tree ('vollstaendiger Baum') by the method described in
    Mathis, S.: Permanentmagneterregte Line-Start-Antriebe in Ferrittechnik,
    PhD thesis, TU Kaiserslautern, Shaker-Verlag, 2019, p.61f.

    A deep copy of the graph is created and the algorithmus tries to remove edges while keeping all nodes connected.
    If a connection would be lost due to the removal of the current edge, keep the edge instead and try removing the next one.
    The returned result of this operation is a reduced network which connects all nodes w/o having a circular connection.

    A traditional problem of mesh analysis has been the treatment of current sources. This implementation uses the following strategy to deal with those:
    When building the spanning tree, two consecutive passes are done over all edges. In the first pass, only branches with current sources are tried.
    If they can be made into an independent branch, they automatically define the mesh current through this branch. If they can't be made into an
    independent branch, the network is overdefined. They are then ignored during the solving process and only used in a post-processing step.
    In the second pass, all non-current-source branches are tried again until a spanning tree remains.
    */
    let mut spanning_tree: StableUnGraph<usize, Type> = graph.clone().into();

    // Storage for the edge indices of the independent branches. An independent branch may contain more than one edge.
    let mut independent_branches: Vec<IndependentBranch> = Vec::new();

    // First pass: Only consider current sources
    for edge in graph.edge_references() {
        if edge.weight() == &Type::Current {
            try_create_new_mesh(
                graph,
                &mut spanning_tree,
                &mut independent_branches,
                edge.id(),
            );
        }
    }

    // All meshes which have been created until now are current source meshes, which have to be treated in a special way.
    let number_current_source_meshes = independent_branches.len();

    // Second pass: Ignore all current sources
    for edge in graph.edge_references() {
        if edge.weight() != &Type::Current {
            try_create_new_mesh(
                graph,
                &mut spanning_tree,
                &mut independent_branches,
                edge.id(),
            );
        }
    }
    return SpanningTree {
        graph: spanning_tree,
        independent_branches,
        number_current_source_meshes,
    };
}

fn slices_are_identical<T: PartialEq>(va: &[T], vb: &[T]) -> bool {
    (va.len() == vb.len()) &&  // zip stops at the shortest
     va.iter()
       .zip(vb)
       .all(|(a,b)| *a == *b)
}

fn update_edge_by_node(edge: &mut NodeEdge, node_tag: usize, side_index: usize) {
    if side_index == 0 {
        edge.source = node_tag;
    } else {
        edge.target = node_tag;
    }
}
