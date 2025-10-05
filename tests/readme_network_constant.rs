mod readme_network;
use readme_network::*;

use network_analysis::*;

#[test]
fn test_resistances_single_current_source() {
    fn inner<N: NetworkAnalysis>() {
        let network = network_creation(false, true);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Slice(&[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
                CurrentSources::Slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                VoltageSources::None,
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), 0);

        let c = solution.currents();
        approx::assert_abs_diff_eq!(c[0], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], 0.6, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], 0.4, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], 0.2, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], -0.2, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], -0.2, epsilon = 0.001);

        let v = solution.voltages();
        approx::assert_abs_diff_eq!(v[0], -1.6, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[1], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[2], 0.6, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[3], 0.4, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[4], 0.2, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[5], 0.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[6], -0.2, epsilon = 0.001);

        current_sum_check(&solution);
        voltage_sum_check(&solution);
    }

    inner::<MeshAnalysis>();
    inner::<NodalAnalysis>();
}

#[test]
fn test_resistances_multiple_current_sources() {
    fn inner<N: NetworkAnalysis>() {
        let network = network_creation(false, false);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Slice(&[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
                CurrentSources::Slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
                VoltageSources::None,
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), 0);

        let c = solution.currents();
        approx::assert_abs_diff_eq!(c[0], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], 1.333, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], -0.333, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], 1.666, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], 2.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], 2.0, epsilon = 0.001);

        // Since the resistances are 1, the voltages are equal to the currents (except at the voltage sources)
        let v = solution.voltages();
        approx::assert_abs_diff_eq!(v[0], -2.333, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[1], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[2], 1.333, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[3], -0.333, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[4], 1.666, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[5], -3.666, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[6], 2.0, epsilon = 0.001);

        current_sum_check(&solution);
        voltage_sum_check(&solution);
    }

    inner::<MeshAnalysis>();
    inner::<NodalAnalysis>();
}

#[test]
fn test_resistances_single_voltage_source() {
    fn inner<N: NetworkAnalysis>() {
        let network = network_creation(true, true);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Slice(&[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
                CurrentSources::None,
                VoltageSources::Slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), 0);

        let c = solution.currents();
        approx::assert_abs_diff_eq!(c[0], -0.625, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], -0.625, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], -0.375, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], -0.25, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], -0.125, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], 0.125, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], 0.125, epsilon = 0.001);

        // Since the resistances are 1, the voltages are equal to the currents (except at the voltage sources)
        let v = solution.voltages();
        approx::assert_abs_diff_eq!(v[0], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[1], -0.625, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[2], -0.375, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[3], -0.25, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[4], -0.125, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[5], 0.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[6], 0.125, epsilon = 0.001);

        current_sum_check(&solution);
        voltage_sum_check(&solution);
    }

    inner::<MeshAnalysis>();
    inner::<NodalAnalysis>();
}

#[test]
fn test_resistances_multiple_voltage_sources() {
    fn inner_p1v_p1v<N: NetworkAnalysis>() {
        let network = network_creation(true, true);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Slice(&[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
                CurrentSources::None,
                VoltageSources::Slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), 0);

        let c = solution.currents();
        approx::assert_abs_diff_eq!(c[0], -0.5, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], -0.5, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], -0.5, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], 0.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], -0.5, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], -0.5, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], -0.5, epsilon = 0.001);

        // Since the resistances are 1, the voltages are equal to the currents (except at the voltage sources)
        let v = solution.voltages();
        approx::assert_abs_diff_eq!(v[0], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[1], -0.5, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[2], -0.5, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[3], 0.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[4], -0.5, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[5], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[6], -0.5, epsilon = 0.001);

        current_sum_check(&solution);
        voltage_sum_check(&solution);
    }

    fn inner_p1v_p2v<N: NetworkAnalysis>() {
        let network = network_creation(true, true);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Slice(&[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
                CurrentSources::None,
                VoltageSources::Slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), 0);

        let c = solution.currents();
        approx::assert_abs_diff_eq!(c[0], -0.375, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], -0.375, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], -0.625, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], 0.25, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], -0.875, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], -1.125, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], -1.125, epsilon = 0.001);

        // Since the resistances are 1, the voltages are equal to the currents (except at the voltage sources)
        let v = solution.voltages();
        approx::assert_abs_diff_eq!(v[0], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[1], -0.375, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[2], -0.625, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[3], 0.25, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[4], -0.875, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[5], 2.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[6], -1.125, epsilon = 0.001);

        current_sum_check(&solution);
        voltage_sum_check(&solution);
    }

    fn inner_n2v_p4v<N: NetworkAnalysis>() {
        let network = network_creation(true, true);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Slice(&[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
                CurrentSources::None,
                VoltageSources::Slice(&[-2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0]),
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), 0);

        let c = solution.currents();
        approx::assert_abs_diff_eq!(c[0], 1.75, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], 1.75, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], 0.25, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], 1.5, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], -1.25, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], -2.75, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], -2.75, epsilon = 0.001);

        // Since the resistances are 1, the voltages are equal to the currents (except at the voltage sources)
        let v = solution.voltages();
        approx::assert_abs_diff_eq!(v[0], -2.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[1], 1.75, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[2], 0.25, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[3], 1.5, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[4], -1.25, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[5], 4.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[6], -2.75, epsilon = 0.001);

        current_sum_check(&solution);
        voltage_sum_check(&solution);
    }

    inner_p1v_p1v::<MeshAnalysis>();
    inner_p1v_p2v::<MeshAnalysis>();
    inner_n2v_p4v::<MeshAnalysis>();
    inner_p1v_p1v::<NodalAnalysis>();
    inner_p1v_p2v::<NodalAnalysis>();
    inner_n2v_p4v::<NodalAnalysis>();
}

#[test]
fn test_resistances_current_and_voltage_sources() {
    fn inner<N: NetworkAnalysis>() {
        let network = network_creation(true, false);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Slice(&[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
                CurrentSources::Slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0]),
                VoltageSources::Slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), 0);

        let c = solution.currents();
        approx::assert_abs_diff_eq!(c[0], -1.6, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], -1.6, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], 0.6, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], -2.2, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], 2.8, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], 5.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], 5.0, epsilon = 0.001);

        // Since the resistances are 1, the voltages are equal to the currents (except at the voltage sources)
        let v = solution.voltages();
        approx::assert_abs_diff_eq!(v[0], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[1], -1.6, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[2], 0.6, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[3], -2.2, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[4], 2.8, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[5], -7.8, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[6], 5.0, epsilon = 0.001);

        current_sum_check(&solution);
        voltage_sum_check(&solution);
    }

    inner::<MeshAnalysis>();
    inner::<NodalAnalysis>();
}
