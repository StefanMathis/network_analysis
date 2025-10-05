use network_analysis::*;

/// Example network from the README.md (image see doc/example.svg)
/// The excitations can be adjusted via the `exc` input parameter.
pub fn network_creation(first_exc_is_voltage: bool, second_exc_is_voltage: bool) -> Network {
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

pub fn current_sum_check(solution: &Solution) {
    let c = solution.currents();

    /*
    According to Kirchhoff's current law, the sum of the currents going into and out of a node must be zero.
    0 = -i0 + i1 + i2 - i4
    0 = i0 - i2 - i3
    0 = -i1 + i3 + i4
    */
    approx::assert_abs_diff_eq!(0.0, -c[1] + c[2] + c[4] - c[5], epsilon = 1e-3);
    approx::assert_abs_diff_eq!(0.0, c[0] - c[2] - c[3], epsilon = 1e-3);
    approx::assert_abs_diff_eq!(0.0, c[3] - c[4] + c[6], epsilon = 1e-3);
}

pub fn voltage_sum_check(solution: &Solution) {
    let v = solution.voltages();

    /*
    According to Kirchhoff's voltage law, the sum of all voltages of any loop must be zero
    0 = v0 + v1 + v2
    0 = v0 + v3 - v6 - v5 + v1
    0 = v0 + v3 + v4 + v1
    0 = v2 + v5 + v6 - v3
    0 = v2 - v4 - v3
    0 = v4 + v5 + v6
     */
    approx::assert_abs_diff_eq!(0.0, v[0] + v[1] + v[2], epsilon = 1e-3);
    approx::assert_abs_diff_eq!(0.0, v[0] + v[3] - v[6] - v[5] + v[1], epsilon = 1e-3);
    approx::assert_abs_diff_eq!(0.0, v[0] + v[3] + v[4] + v[1], epsilon = 1e-3);
    approx::assert_abs_diff_eq!(0.0, v[2] + v[5] + v[6] - v[3], epsilon = 1e-3);
    approx::assert_abs_diff_eq!(0.0, v[2] - v[4] - v[3], epsilon = 1e-3);
    approx::assert_abs_diff_eq!(0.0, v[4] + v[5] + v[6], epsilon = 1e-3);
}
