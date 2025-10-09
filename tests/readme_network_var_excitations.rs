mod readme_network;
use readme_network::*;

use network_analysis::*;

fn calc_voltage_exc<'a>(mut args: FunctionArgs) {
    // Voltage inverse proportional to square root of absolute current, limited to 10 V max
    for (source, (idx, exc)) in args.edge_value_and_type.iter_mut().enumerate() {
        let curr = args.edge_currents[idx].abs();
        if curr < 0.1 {
            *exc = (source as f64 + 1.0) * 10.0;
        } else {
            *exc = (source as f64 + 1.0) / curr.sqrt();
        }
    }
}

#[test]
fn test_two_voltage_sources() {
    fn inner<N: NetworkAnalysis>(expected_iter_count: usize) {
        let network = network_creation(true, true);

        let mut network_analysis = N::new(&network);

        let config = SolverConfig::default();
        let solution = network_analysis
            .solve(
                Resistances::Slice(&[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
                CurrentSources::None(PhantomData),
                VoltageSources::Function(&calc_voltage_exc),
                None,
                None,
                None,
                &config,
            )
            .unwrap();

        assert_eq!(solution.iter_count(), expected_iter_count);

        let c = solution.currents();
        approx::assert_abs_diff_eq!(c[0], -0.578, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[1], -0.578, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[2], -0.737, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[3], 0.158, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[4], -0.895, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[5], -1.053, epsilon = 0.001);
        approx::assert_abs_diff_eq!(c[6], -1.053, epsilon = 0.001);

        let v = solution.voltages();
        approx::assert_abs_diff_eq!(v[0], 1.315, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[1], -0.578, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[2], -0.737, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[3], 0.158, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[4], -0.895, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[5], 1.949, epsilon = 0.001);
        approx::assert_abs_diff_eq!(v[6], -1.053, epsilon = 0.001);

        let r = solution.resistances();
        approx::assert_abs_diff_eq!(r[0], 0.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(r[1], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(r[2], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(r[3], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(r[4], 1.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(r[5], 0.0, epsilon = 0.001);
        approx::assert_abs_diff_eq!(r[6], 1.0, epsilon = 0.001);

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

    inner::<MeshAnalysis>(76);
    inner::<NodalAnalysis>(74);
}
