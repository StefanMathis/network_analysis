use network_analysis::*;
use petgraph::prelude::*;

/// A very simple network which only consists of three parallel resistors
#[test]
fn test_three_parallel_resistors() -> () {
    // From graph
    let mut graph = UnGraph::<usize, Type>::with_capacity(2, 3);
    graph.add_node(0);
    graph.add_node(1);
    graph.add_edge(0.into(), 1.into(), Type::Resistance);
    graph.add_edge(0.into(), 1.into(), Type::Resistance);
    graph.add_edge(0.into(), 1.into(), Type::Resistance);
    assert!(Network::new(graph).is_ok());

    // From NodeEdge
    let edges = [
        NodeEdge::new(0, 1, Type::Resistance),
        NodeEdge::new(0, 1, Type::Resistance),
        NodeEdge::new(0, 1, Type::Resistance),
    ];
    assert!(Network::from_node_edges(edges.as_slice()).is_ok());

    // From EdgeListEdge
    let edges = [
        EdgeListEdge::new(vec![1, 2], vec![1, 2], Type::Resistance),
        EdgeListEdge::new(vec![0, 2], vec![0, 2], Type::Resistance),
        EdgeListEdge::new(vec![0, 1], vec![0, 1], Type::Resistance),
    ];
    assert!(Network::from_edge_list_edges(edges.as_slice()).is_ok());
}

/**
This network is not solvable except for the trivial case of the voltage sources at
edge 0 and edge 2 having the same excitation value.
 */
#[test]
fn test_two_parallel_voltage_sources() -> () {
    // From EdgeListEdge
    let edges = [
        EdgeListEdge::new(vec![1, 2], vec![1, 2], Type::Voltage),
        EdgeListEdge::new(vec![0, 2], vec![0, 2], Type::Resistance),
        EdgeListEdge::new(vec![0, 1], vec![0, 1], Type::Voltage),
    ];
    assert!(Network::from_edge_list_edges(edges.as_slice()).is_err());
}

/**
This network is not solvable except for the trivial case of all three voltage sources
having the same excitation value.
 */
#[test]
fn test_three_parallel_voltage_sources() -> () {
    // From EdgeListEdge
    let edges = [
        EdgeListEdge::new(vec![1, 2], vec![1, 2], Type::Voltage),
        EdgeListEdge::new(vec![0, 2], vec![0, 2], Type::Voltage),
        EdgeListEdge::new(vec![0, 1], vec![0, 1], Type::Voltage),
    ];
    assert!(Network::from_edge_list_edges(edges.as_slice()).is_err());
}

/**
This network cannot be solved, if `2`, `3` and `4` all are current sources
(except for the trivial case where the values add up to zero, but in this
case one of the sources can be omitted anyway). Therefore, network generation
fails.

```text
 ┌──[0]──┬──[1]──┐
 │       │       │
[2]     [3]     [4]
 │       │       │
 └──[5]──┴──[6]──┘
```
 */
#[test]
fn test_contradictory_current_sources() {
    let edges = [
        EdgeListEdge::new(vec![2], vec![1, 3], Type::Resistance),
        EdgeListEdge::new(vec![0, 3], vec![4], Type::Resistance),
        EdgeListEdge::new(vec![0], vec![5], Type::Current),
        EdgeListEdge::new(vec![0, 1], vec![5, 6], Type::Current),
        EdgeListEdge::new(vec![1], vec![6], Type::Current),
        EdgeListEdge::new(vec![2], vec![3, 6], Type::Resistance),
        EdgeListEdge::new(vec![3, 5], vec![4], Type::Resistance),
    ];
    let err = Network::from_edge_list_edges(edges.as_slice()).unwrap_err();
    match err {
        BuildError::OverdefinedByCurrentSources { edge, graph: _ } => assert_eq!(edge.index(), 4),
        _ => unreachable!(),
    };
}

/**
This network cannot be solved, if `0`, `3`, `5` and `2` all are voltage sources
(except for the trivial case where the values add up to zero, but in this
case one of the sources can be omitted anyway). Therefore, network generation
fails.

```text
 ┌──[0]──┬──[1]──┐
 │       │       │
[2]     [3]     [4]
 │       │       │
 └──[5]──┴──[6]──┘
```
 */
#[test]
fn test_contradictory_voltage_sources() {
    let edges = [
        EdgeListEdge::new(vec![2], vec![1, 3], Type::Voltage),
        EdgeListEdge::new(vec![0, 3], vec![4], Type::Resistance),
        EdgeListEdge::new(vec![0], vec![5], Type::Voltage),
        EdgeListEdge::new(vec![0, 1], vec![5, 6], Type::Voltage),
        EdgeListEdge::new(vec![1], vec![6], Type::Resistance),
        EdgeListEdge::new(vec![2], vec![3, 6], Type::Voltage),
        EdgeListEdge::new(vec![3, 5], vec![4], Type::Resistance),
    ];
    let err = Network::from_edge_list_edges(edges.as_slice()).unwrap_err();
    match err {
        BuildError::OverdefinedByVoltageSources { edge: _, graph: _ } => (),
        _ => unreachable!(),
    };
}

/**
This network has a dead end:

```text
 ──[0]──┬──[1]──┐
        │       │
       [2]     [3]
        │       │
        └───────┘
```
 */
#[test]
fn test_dead_end() {
    let mut branches: Vec<EdgeListEdge> = Vec::new();

    // "Dangling" edge
    branches.push(EdgeListEdge::new(vec![], vec![1, 2], Type::Resistance));
    branches.push(EdgeListEdge::new(vec![0, 2], vec![3], Type::Resistance));
    branches.push(EdgeListEdge::new(vec![0, 1], vec![3], Type::Resistance));
    branches.push(EdgeListEdge::new(vec![1], vec![2], Type::Resistance));

    let err = Network::from_edge_list_edges(&branches).unwrap_err();
    match err {
        BuildError::DeadEnd { node, graph: _ } => assert_eq!(node.index(), 0),
        _ => unreachable!("test failed"),
    }
}

#[test]
fn test_multiple_current_sources_in_path() {
    {
        let mut branches: Vec<EdgeListEdge> = Vec::new();
        branches.push(EdgeListEdge::new(vec![2, 4, 5], vec![1], Type::Current));
        branches.push(EdgeListEdge::new(vec![2, 3], vec![0], Type::Current));
        branches.push(EdgeListEdge::new(
            vec![1, 3],
            vec![0, 4, 5],
            Type::Resistance,
        ));
        branches.push(EdgeListEdge::new(vec![1, 2], vec![4, 5], Type::Resistance));
        branches.push(EdgeListEdge::new(
            vec![3, 5],
            vec![0, 2, 5],
            Type::Resistance,
        ));
        branches.push(EdgeListEdge::new(
            vec![3, 4],
            vec![0, 2, 4],
            Type::Resistance,
        ));
        assert!(Network::from_edge_list_edges(&branches).is_err());
    }
    {
        // Single mesh with two current sources
        let mut branches: Vec<EdgeListEdge> = Vec::new();
        branches.push(EdgeListEdge::new(vec![2], vec![1], Type::Resistance));
        branches.push(EdgeListEdge::new(vec![0], vec![2], Type::Current));
        branches.push(EdgeListEdge::new(vec![1], vec![0], Type::Current));
        assert!(Network::from_edge_list_edges(&branches).is_err());
    }
}
