use network_analysis::*;

/// A very simple network which only consists of three parallel resistors
fn example_network_creation(exc: [Type; 3]) -> Network {
    let edges = [
        NodeEdge::new(0, 1, exc[0]),
        NodeEdge::new(0, 1, exc[1]),
        NodeEdge::new(0, 1, exc[2]),
    ];
    return Network::from_node_edges(edges.as_slice()).expect("this is a valid network");
}

fn example_network_sum_check(solution: &Solution<'_>) {
    let c = solution.currents();
    approx::assert_abs_diff_eq!(0.0, c[0] + c[1] + c[2], epsilon = 1e-3);
}

#[test]
fn test_constant_resistances_single_constant_voltage_source() {
    fn inner<N: NetworkAnalysis>() {
        let network = example_network_creation([Type::Voltage, Type::Resistance, Type::Resistance]);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Slice(&[0.0, 1.0, 1.0]),
                CurrentSources::None,
                VoltageSources::Slice(&[1.0, 0.0, 0.0]),
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), 0);

        approx::assert_abs_diff_eq!(solution.currents()[0], -2.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(solution.currents()[1], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(solution.currents()[2], 1.0, epsilon = 0.001);

        // Since the resistances are 1, the voltages are equal to the currents
        approx::assert_abs_diff_eq!(solution.voltages()[0], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(solution.voltages()[1], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(solution.voltages()[2], 1.0, epsilon = 0.001);

        example_network_sum_check(&solution);
    }

    inner::<MeshAnalysis>();
    inner::<NodalAnalysis>();
}

#[test]
fn test_constant_resistances_single_constant_current_source() {
    fn inner<N: NetworkAnalysis>() {
        let network = example_network_creation([Type::Current, Type::Resistance, Type::Resistance]);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                EdgeValueInputs::Slice(&[0.0, 1.0, 1.0]),
                EdgeValueInputs::Slice(&[1.0, 0.0, 0.0]),
                EdgeValueInputs::None,
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), 0);

        approx::assert_abs_diff_eq!(solution.currents()[0], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(solution.currents()[1], -0.5, epsilon = 0.001);
        approx::assert_abs_diff_eq!(solution.currents()[2], -0.5, epsilon = 0.001);

        // Since the resistances are 1, the voltages are equal to the currents
        approx::assert_abs_diff_eq!(solution.voltages()[0], -0.5, epsilon = 0.001);
        approx::assert_abs_diff_eq!(solution.voltages()[1], -0.5, epsilon = 0.001);
        approx::assert_abs_diff_eq!(solution.voltages()[2], -0.5, epsilon = 0.001);

        example_network_sum_check(&solution);
    }

    inner::<MeshAnalysis>();
    inner::<NodalAnalysis>();
}

#[test]
fn test_constant_resistances_two_constant_current_sources() {
    fn inner<N: NetworkAnalysis>() {
        let network = example_network_creation([Type::Current, Type::Resistance, Type::Current]);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                EdgeValueInputs::Slice(&[0.0, 1.0, 0.0]),
                EdgeValueInputs::Slice(&[1.0, 0.0, 1.0]),
                EdgeValueInputs::None,
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), 0);

        approx::assert_abs_diff_eq!(solution.currents()[0], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(solution.currents()[1], -2.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(solution.currents()[2], 1.0, epsilon = 0.001);

        // Since the resistances are 1, the voltages are equal to the currents
        approx::assert_abs_diff_eq!(solution.voltages()[0], -2.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(solution.voltages()[1], -2.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(solution.voltages()[2], -2.0, epsilon = 0.001);

        example_network_sum_check(&solution);
    }

    inner::<MeshAnalysis>();
    inner::<NodalAnalysis>();
}
