use network_analysis::{
    CurrentSources, MeshAnalysis, Network, NetworkAnalysis, NodalAnalysis, Resistances, Solution,
    SolverConfig, Type, VoltageSources,
};
use petgraph::graph::UnGraph;

/// The graph looks like this (petgraph uses corner numbers):
/// ```
///      2V
/// 0 -- A -- 1
/// |         |
/// D 6V      B 2V
/// |         |
/// 3 -- C -- 2
///      2V
/// ```
fn petgraph_network() -> Network {
    let g = UnGraph::<usize, Type>::from_edges([
        (0, 1, Type::Resistance),
        (1, 2, Type::Resistance),
        (2, 3, Type::Resistance),
        (3, 0, Type::Voltage),
    ]);
    Network::new(g).expect("valid network")
}

#[test]
fn test_circuit_petgraph_mesh_analysis() {
    let network = petgraph_network();
    let mut analyser = MeshAnalysis::new(&network);
    let solution = analyze(&mut analyser);
    verify_circuit(&solution);
}

#[test]
fn test_circuit_petgraph_nodal_analysis() {
    let network = petgraph_network();
    let mut analyser = NodalAnalysis::new(&network);
    let solution = analyze(&mut analyser);
    verify_circuit(&solution);
}

fn verify_circuit(solution: &Solution<'_>) {
    let v = &solution.voltages();
    let c = &solution.currents();

    // Kirchhoff's voltage law, sum of voltages must be zero.
    approx::assert_abs_diff_eq!(0.0, v[0] + v[1] + v[2] + v[3], epsilon = 1e-3);

    // Simple loop, current flow is equal in direction all the way around.
    approx::assert_abs_diff_eq!(c[0], -0.002, epsilon = 1e-3);
    approx::assert_abs_diff_eq!(c[1], -0.002, epsilon = 1e-3);
    approx::assert_abs_diff_eq!(c[2], -0.002, epsilon = 1e-3);
    approx::assert_abs_diff_eq!(c[3], -0.002, epsilon = 1e-3);

    // 3 resistors take 1/3 of 6 volts = 2 volts.
    approx::assert_abs_diff_eq!(v[0], -2.0, epsilon = 1e-3);
    approx::assert_abs_diff_eq!(v[1], -2.0, epsilon = 1e-3);
    approx::assert_abs_diff_eq!(v[2], -2.0, epsilon = 1e-3);
    approx::assert_abs_diff_eq!(v[3], 6.0, epsilon = 1e-3);
}

fn analyze<T: NetworkAnalysis>(analyzer: &mut T) -> Solution<'_> {
    let current_exc = CurrentSources::none();
    let voltage_src = VoltageSources::Slice(&[0.0, 0.0, 0.0, 6.0]);
    let resistances = Resistances::Slice(&[1000.0, 1000.0, 1000.0, 0.0]);
    let config = SolverConfig::default();

    analyzer
        .solve(
            resistances,
            current_exc,
            voltage_src,
            None,
            None,
            None,
            &config,
        )
        .expect("solvable")
}
