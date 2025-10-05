mod readme_network;
use readme_network::*;

use network_analysis::*;

use rayon::prelude::*;

fn calc_resistances<'a>(mut input: FunctionArgs<'_>) {
    input
        .edge_value_and_type
        .par_iter_mut()
        .for_each(|(idx, resistance)| {
            *resistance = 1.0 + input.edge_currents[idx].abs().sqrt();
        });
}

#[test]
fn test_resistances_single_current_source() {
    fn inner<N: NetworkAnalysis>(expected_iter_count: usize) {
        let network = network_creation(false, true);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Function(&calc_resistances),
                CurrentSources::Slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                VoltageSources::None,
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), expected_iter_count);

        let c = solution.currents();
        approx::assert_abs_diff_eq!(c[0], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], 0.575, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], 0.425, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], 0.212, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], -0.212, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], -0.212, epsilon = 0.001);

        let v = solution.voltages();
        approx::assert_abs_diff_eq!(v[0], -3.012, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[1], 2.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[2], 1.011, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[3], 0.701, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[4], 0.310, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[5], 0.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[6], -0.310, epsilon = 0.001);

        let r = solution.resistances();
        approx::assert_abs_diff_eq!(r[0], 0.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(r[1], 2.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(r[2], 1.758, epsilon = 0.001);
        approx::assert_abs_diff_eq!(r[3], 1.652, epsilon = 0.001);
        approx::assert_abs_diff_eq!(r[4], 1.461, epsilon = 0.001);
        approx::assert_abs_diff_eq!(r[5], 0.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(r[6], 1.461, epsilon = 0.001);

        let i: Vec<usize> = solution
            .resistances_and_indices()
            .iter()
            .map(|(i, _)| i)
            .collect();
        assert_eq!(i[0], 1);
        assert_eq!(i[1], 2);
        assert_eq!(i[2], 3);
        assert_eq!(i[3], 4);
        assert_eq!(i[4], 6);

        current_sum_check(&solution);
        voltage_sum_check(&solution);
    }

    inner::<MeshAnalysis>(10);
    inner::<NodalAnalysis>(9);
}

#[test]
fn test_resistances_multiple_current_sources() {
    fn inner<N: NetworkAnalysis>(expected_iter_count: usize) {
        let network = network_creation(false, false);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Function(&calc_resistances),
                CurrentSources::Slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
                VoltageSources::None,
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), expected_iter_count);

        let c = solution.currents();

        approx::assert_abs_diff_eq!(c[0], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], 1.389, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], -0.388, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], 1.611, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], 2.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], 2.0, epsilon = 0.001);

        current_sum_check(&solution);
        voltage_sum_check(&solution);
    }

    inner::<MeshAnalysis>(13);
    inner::<NodalAnalysis>(11);
}

#[test]
fn test_resistances_single_voltage_source() {
    fn inner<N: NetworkAnalysis>(expected_iter_count: usize) {
        let network = network_creation(true, true);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Function(&calc_resistances),
                CurrentSources::None,
                VoltageSources::Slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), expected_iter_count);

        let c = solution.currents();

        approx::assert_abs_diff_eq!(c[0], -0.401, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], -0.401, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], -0.233, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], -0.168, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], -0.084, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], 0.084, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], 0.084, epsilon = 0.001);

        current_sum_check(&solution);
        voltage_sum_check(&solution);
    }

    inner::<MeshAnalysis>(11);
    inner::<NodalAnalysis>(10);
}

#[test]
fn test_resistances_multiple_voltage_sources() {
    fn inner_p1v_p1v<N: NetworkAnalysis>(expected_iter_count: usize) {
        let network = network_creation(true, true);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Function(&calc_resistances),
                CurrentSources::None,
                VoltageSources::Slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), expected_iter_count);

        let c = solution.currents();
        approx::assert_abs_diff_eq!(c[0], -0.319, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], -0.319, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], -0.319, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], 0.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], -0.319, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], -0.319, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], -0.319, epsilon = 0.001);

        current_sum_check(&solution);
        voltage_sum_check(&solution);
    }

    fn inner_p1v_p2v<N: NetworkAnalysis>(expected_iter_count: usize) {
        let network = network_creation(true, true);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Function(&calc_resistances),
                CurrentSources::None,
                VoltageSources::Slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), expected_iter_count);

        let c = solution.currents();
        approx::assert_abs_diff_eq!(c[0], -0.243, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], -0.243, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], -0.392, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], 0.148, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], -0.494, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], -0.643, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], -0.643, epsilon = 0.001);

        current_sum_check(&solution);
        voltage_sum_check(&solution);
    }

    fn inner_n2v_p4v<N: NetworkAnalysis>(expected_iter_count: usize) {
        let network = network_creation(true, true);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Function(&calc_resistances),
                CurrentSources::None,
                VoltageSources::Slice(&[-2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0]),
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), expected_iter_count);

        let c = solution.currents();
        approx::assert_abs_diff_eq!(c[0], 0.9, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], 0.9, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], 0.173, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], 0.727, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], -0.616, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], -1.343, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], -1.343, epsilon = 0.001);

        current_sum_check(&solution);
        voltage_sum_check(&solution);
    }

    inner_p1v_p1v::<MeshAnalysis>(11);
    inner_p1v_p2v::<MeshAnalysis>(12);
    inner_n2v_p4v::<MeshAnalysis>(15);

    inner_p1v_p1v::<NodalAnalysis>(10);
    inner_p1v_p2v::<NodalAnalysis>(12);
    inner_n2v_p4v::<NodalAnalysis>(14);
}

#[test]
fn test_resistances_current_and_voltage_sources() {
    fn inner<N: NetworkAnalysis>(expected_iter_count: usize) {
        let network = network_creation(true, false);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Function(&calc_resistances),
                CurrentSources::Slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0]),
                VoltageSources::Slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), expected_iter_count);

        let c = solution.currents();
        approx::assert_abs_diff_eq!(c[0], -1.310, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], -1.310, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], 0.922, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], -2.232, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], 2.768, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], 5.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], 5.0, epsilon = 0.001);

        current_sum_check(&solution);
        voltage_sum_check(&solution);
    }

    inner::<MeshAnalysis>(14);
    inner::<NodalAnalysis>(13);
}

#[test]
fn test_pow2() {
    fn calc_resistances_pow2<'a>(mut input: FunctionArgs<'_>) {
        input
            .edge_value_and_type
            .par_iter_mut()
            .for_each(|(idx, resistance)| {
                *resistance = 1.0 + input.edge_currents[idx].powi(2);
            });
    }

    /*
    This creates the following network with two voltage sources at 0 and 6
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
    edges.push(EdgeListEdge::new(vec![1, 2], vec![4, 5], Type::Voltage));
    let network = Network::from_edge_list_edges(&edges).expect("valid network");

    let mut mesh_analysis = MeshAnalysis::new(&network);

    let sol = mesh_analysis
        .solve(
            Resistances::Function(&mut calc_resistances_pow2),
            CurrentSources::None,
            VoltageSources::Slice(&[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            None,
            None,
            None,
            &Default::default(),
        )
        .expect("can be solved");
    assert_eq!(sol.iter_count(), 119);
}
