/*!
Shared network analysis components.

This module contains the following elements:
- [`NetworkAnalysis`], a trait implemented for both [`MeshAnalysis`](crate::MeshAnalysis) and
[`NodalAnalysis`](crate::NodalAnalysis) which specifies a common interface for both construction
and solving as well as the corresponding [`SolveError`].
- [`SolverConfig`], a struct which defines the solver parameters for the underlying Newton-Raphson algorithm.
- [`EdgeValueInputs`], an enum defining either constant or variable resistances, edge current / voltage sources as well
as three aliases [`Resistances`], [`CurrentSources`] and [`VoltageSources`].
*/

use crate::{Type, finite_diff::central_jacobian, network::Network};
use approx::ulps_eq;
use na::{DMatrix, DVector};
use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Constants
const RESISTANCE: u8 = 0;
const CURRENT: u8 = 1;
const VOLTAGE: u8 = 2;
pub(crate) const MESH_ANALYSIS: u8 = 0;
pub(crate) const NODAL_ANALYSIS: u8 = 1;

/**
An error returned from a failed call to [`solve`](`NetworkAnalysis::solve`).
 */
#[derive(Debug, Clone)]
pub enum SolveError {
    /**
    The maximum number of iterations defined in [`SolverConfig`] has been reached
    without finding a solution. Consider either increasing the iteration maximum or relaxing the tolerances
    (see docstring of [`SolverConfig`]).
    */
    MaxIterReached(usize),
    /**
    The Jacobian has become singular. This can e.g. happen due to variable resistances / inductances
    becoming (almost) zero. Make sure the provided [`EdgeValueInputs`] does return reasonable values for all inputs (e.g. by
    clamping) and that the custom Jacobian function (if one is given) only returns finite, non-nan entries for the Jacobian.
     */
    SingularJacobian,
    /**
    For one of resistance, current or voltage sources (see `slice_type`), the `Slice` variant has been used.
    The length of the underlying slice is not equal to the edge count.
     */
    SliceWrongLength {
        slice_type: Type,
        slice_length: usize,
        edge_count: usize,
    },
    /**
    For one of resistance, current or voltage sources (see `idx_and_val_type`), the `IdxAndVals` variant has been used.
    At least one of the indices (`idx`) is equal to or larger than the edge count, which would result in an out-of-bounds access.
    */
    OutOfBoundsIdx {
        idx_and_val_type: Type,
        idx: usize,
        edge_count: usize,
    },
    /**
    [`Resistances`] must always have positive values (larger than zero), but the negative value `val` was found at edge `idx`.
     */
    NonPositiveResistance { val: f64, idx: usize },
    /**
    If one of [`Resistances`], [`CurrentSources`] or [`VoltageSources`] produce a nonzero value `val` at edge `idx`,
    but the edge has a different type `edge_type_at_idx`, this error variant is returned. As explained in the docstrings
    of the two enums, they must only produce nonzero values for edges of the same type.
     */
    NonzeroForMismatchedEdgeType {
        val: f64,
        idx: usize,
        edge_type_at_idx: Type,
        edge_type_check: Type,
    },
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveError::MaxIterReached(count) => write!(
                                                f,
                                                "defined maximum number of iterations {count} reached without converging. Try to reduce the convergence
                criteria or to increase the maximum number of iterations (see `SolverConfig`)."
                                            ),
            SolveError::SingularJacobian => write!(f, "resulting jacobian is singular (has nan-entries). Check the resistance calculation functions."),
            SolveError::SliceWrongLength { slice_type, slice_length, edge_count } => {
                        let slice_type_stringified = match slice_type {
                            Type::Voltage => "voltage excitation",
                            Type::Current => "current excitation",
                            Type::Resistance => "resistance",
                        };
                        write!(f, "given {slice_type_stringified} slice length {slice_length} does not match edge count {edge_count}.")
                    },
            SolveError::OutOfBoundsIdx { idx_and_val_type, idx, edge_count } => {
                        let idx_and_val_type_stringified = match idx_and_val_type {
                            Type::Voltage => "voltage excitation",
                            Type::Current => "current excitation",
                            Type::Resistance => "resistance",
                        };
                        write!(f, "index {idx} specified in the {idx_and_val_type_stringified} index - value pair slice is larger than the edge count {edge_count}.")
                    },
            SolveError::NonPositiveResistance{val, idx } => write!(
                                        f, "detected negative resistance value {val} at index {idx}."),
            SolveError::NonzeroForMismatchedEdgeType { val, idx, edge_type_at_idx, edge_type_check } => write!(
                                        f, "nonzero value {val} at index {idx} not allowed (expected edge type {edge_type_check}, found {edge_type_at_idx})."),
        }
    }
}

impl std::error::Error for SolveError {}

/**
Solution created by successfull [`solve`](NetworkAnalysis::solve) call

This struct is created after an successfull call to solve and provides
accessors to the calculated edge resistance, edge voltage and edge
current (as slices) as well as the number of iterations. It borrows the edge information
from internal buffers of the analysis struct ([`MeshAnalysis`](crate::MeshAnalysis)
or [`NodalAnalysis`](crate::NodalAnalysis)), therefore it needs to be dropped
before the next call to [`solve`](NetworkAnalysis::solve). This is enforced
by the lifetime constraint. Consider copying the information from
the slices returned by the accessor methods if it needs to persist.
 */
pub struct Solution<'a> {
    buf: &'a Buffer,
    edge_types: &'a [Type],
    iter_count: usize,
}

impl<'a> Solution<'a> {
    pub(crate) fn new(buf: &'a Buffer, edge_types: &'a [Type], iter_count: usize) -> Self {
        Self {
            buf,
            edge_types,
            iter_count,
        }
    }

    /**
    This function returns a filtered view on [`Solution::resistances`], which shows only
    those entries of the edge resistance slice where the corresponding edge is a resistance.
     */
    pub fn resistances_and_indices(&'a self) -> EdgeValueAndType<'a> {
        return EdgeValueAndType {
            values: self.buf.edge_resistances.as_slice(),
            edge_types: self.edge_types,
            value_type: Type::Resistance,
        };
    }

    /**
    Returns all edge resistances of the network.
    If an edge is not a resistance, the corresponding entry in the slice is zero.
     */
    pub fn resistances(&self) -> &[f64] {
        return self.buf.edge_resistances.as_slice();
    }

    /**
    Returns all edge currents of the network.
     */
    pub fn currents(&self) -> &[f64] {
        return self.buf.edge_currents.as_slice();
    }

    /**
    Returns all edge voltages of the network.
     */
    pub fn voltages(&self) -> &[f64] {
        // Add the voltage excitations
        return self.buf.edge_voltages.as_slice();
    }

    /**
    Returns the edge types of the network.
     */
    pub fn edge_types(&self) -> &[Type] {
        // Add the voltage excitations
        return self.edge_types;
    }

    /**
    Returns the number of iterations used during solving.
     */
    pub fn iter_count(&self) -> usize {
        return self.iter_count;
    }
}

/**
This struct contains the edge `values` (resistance, current or voltage) specified by its field `value_type`.
It is usually created from the [`Solution::resistances_and_indices`] method, but can also be constructuted
manually if needed.

The struct methods provide iterators ([`EdgeValueAndType::iter`] and the parallelized
variant [`EdgeValueAndType::par_iter`]) which return only those entries of `values`
where the edge type corresponds to `value_type` as well as the corresponding edge index.
See [`EdgeValueAndType::iter`] for an example.
 */
#[derive(Debug)]
pub struct EdgeValueAndType<'a> {
    pub values: &'a [f64],
    pub edge_types: &'a [Type],
    pub value_type: Type,
}

impl<'a> EdgeValueAndType<'a> {
    /**
    Returns an iterator over the value (resistance, current or voltage) and the corresponding edge index.
    This index is always within bounds of `values` and `edge_types`.

    # Example

    The following entries are the solutions of a network, where the second edge is a voltage source,
    all other edges are resistances. The iterator makes sure only "real" resistance values (belonging
    to actual resistances) are returned.
    ```
    use network_analysis::{Type, EdgeValueAndType};

    let values = &[1.0, 0.0, 3.0, 4.0];
    let edge_types = &[Type::Resistance, Type::Voltage, Type::Resistance, Type::Resistance];
    let edge_value_and_type = EdgeValueAndType {
        values,
        edge_types,
        value_type: Type::Resistance // Filter for resistances
    };

    let mut iter = edge_value_and_type.iter();
    assert_eq!(iter.next(), Some((0, 1.0)));
    assert_eq!(iter.next(), Some((2, 3.0)));
    assert_eq!(iter.next(), Some((3, 4.0)));
    assert_eq!(iter.next(), None);
    ```
     */
    pub fn iter(&self) -> impl Iterator<Item = (usize, f64)> {
        return self
            .values
            .iter()
            .cloned()
            .zip(self.edge_types.iter().cloned())
            .enumerate()
            .filter_map(|(idx, (val, edge))| {
                if edge == self.value_type {
                    Some((idx, val))
                } else {
                    None
                }
            });
    }

    /**
    Parallelized (threaded) variant of [`EdgeValueAndType::iter`] (based on [`rayon`]).
     */
    pub fn par_iter(&self) -> impl ParallelIterator<Item = (usize, f64)> {
        return self
            .values
            .par_iter()
            .cloned()
            .zip(self.edge_types.par_iter().cloned())
            .enumerate()
            .filter_map(|(idx, (val, edge))| {
                if edge == Type::Resistance {
                    Some((idx, val))
                } else {
                    None
                }
            });
    }
}

/**
A variant of [`EdgeValueAndType`] which returns mutable references to the value(s).

This struct is provided as an argument for [`EdgeValueInputs::Function`] or [`EdgeValueInputs::Function`] by [`NetworkAnalysis::solve`]
to give a filtered view on the current / voltage sources (via its iterators [`EdgeValueAndTypeMut::iter_mut`] and the parallelized
variant [`EdgeValueAndTypeMut::par_iter_mut`]). This makes it easier to avoid populating the underlying slices with wrong entries
(e.g. writing a resistance value to a current source edge). The underlying slice is still available in the `values` field.
See [`EdgeValueAndTypeMut::iter_mut`] for an example.
 */
#[derive(Debug)]
pub struct EdgeValueAndTypeMut<'a> {
    pub values: &'a mut [f64],
    pub edge_types: &'a [Type],
    pub value_type: Type,
}

impl<'a> EdgeValueAndTypeMut<'a> {
    /**
    Returns an iterator over a mutable reference of the value (resistance, current or voltage) and the corresponding edge index.
    This index is always within bounds of `values` and `edge_types`.

    # Example

    All resistances of a network are dependent on the current going through them (formula `r = 1 + current^2`).
    Using [`EdgeValueAndTypeMut`], this can be realized in a very simple way.
    ```
    use network_analysis::{Type, EdgeValueAndTypeMut, FunctionArgs};

    fn resistance_calc(mut input: FunctionArgs<'_>) {
        for (idx, resistance) in input.edge_value_and_type.iter_mut() {
            // SAFETY: When using `FunctionArgs`, the indices returned by the iterator are guaranteed to be in bounds of the edge slices.
            let edge_current = unsafe{input.edge_currents.get_unchecked(idx)};
            *resistance = 1.0 + edge_current.powi(2);
        }
    }

    let resistances = &mut [0.0, 0.0, 0.0, 0.0];
    let edge_types = &[Type::Resistance, Type::Voltage, Type::Resistance, Type::Resistance];
    let edge_value_and_type = EdgeValueAndTypeMut {
        values: resistances,
        edge_types,
        value_type: Type::Resistance // Filter for resistances
    };

    // This struct is provided `NetworkAnalysis::solve` when using it for analyzing a network
    let input = FunctionArgs {
        edge_value_and_type,
        edge_currents: &[1.0, 2.0, 3.0, 4.0], // Arbitrary numbers
        edge_voltages: &[1.0, 2.0, 3.0, 4.0], // Arbitrary numbers
    };

    resistance_calc(input);

    // Check the populated values
    assert_eq!(resistances[0], 2.0);
    assert_eq!(resistances[1], 0.0); // Filtered out
    assert_eq!(resistances[2], 10.0);
    assert_eq!(resistances[3], 17.0);
    ```
     */
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &'_ mut f64)> {
        return self
            .values
            .iter_mut()
            .zip(self.edge_types.iter().cloned())
            .enumerate()
            .filter_map(|(idx, (val, edge))| {
                if edge == self.value_type {
                    Some((idx, val))
                } else {
                    None
                }
            });
    }

    /**
    Parallelized (threaded) variant of [`EdgeValueAndTypeMut::iter_mut`] (based on [`rayon`]).
     */
    pub fn par_iter_mut(&mut self) -> impl ParallelIterator<Item = (usize, &'_ mut f64)> {
        return self
            .values
            .par_iter_mut()
            .zip(self.edge_types.par_iter().cloned())
            .enumerate()
            .filter_map(|(idx, (val, edge))| {
                if edge == Type::Resistance {
                    Some((idx, val))
                } else {
                    None
                }
            });
    }
}

/**
A simple container for the values provided to [`EdgeValueInputs::Function`] or [`EdgeValueInputs::Function`] by the [`NetworkAnalysis::solve`] method.
See [`EdgeValueAndTypeMut::iter_mut`] for an example on how to use it.
 */
pub struct FunctionArgs<'a> {
    /// See [`EdgeValueAndTypeMut`].
    pub edge_value_and_type: EdgeValueAndTypeMut<'a>,
    /// Edge currents calculated for the current iteration.
    pub edge_currents: &'a [f64],
    /// Edge voltages calculated for the current iteration.
    pub edge_voltages: &'a [f64],
}

/**
Trait which defines functionality shared between
[`MeshAnalysis`](crate::MeshAnalysis) and [`NodalAnalysis`](crate::NodalAnalysis).
 */
pub trait NetworkAnalysis: Sized {
    /**
    Create a new [`MeshAnalysis`](crate::MeshAnalysis) or [`NodalAnalysis`](crate::NodalAnalysis) instance from the given network.
    Since the network has already been checked during its creation, this operation is infallible.
     */
    fn new(network: &Network) -> Self;

    /**
    Try to solve the network for the given excitations and resistances.

    This is the central method of [`NetworkAnalysis`] which tries to calculate a [`Solution`] which satisfies the given
    constraints (excitations and resistance).
    Depending on the chosen method ([`MeshAnalysis`](crate::MeshAnalysis) or [`NodalAnalysis`](crate::NodalAnalysis)),
    the problem is transformed into a matrix equation of the type `A * x = b` using the [mesh](https://en.wikipedia.org/wiki/Mesh_analysis)
    or [nodal](https://en.wikipedia.org/wiki/Nodal_analysis) methods.
    Such a system can be solved directly if `A` and `b` (resistances and excitations) are constant. If they aren't, the system
    is solved iteratively with the [Newton-Raphson](https://en.wikipedia.org/wiki/Newton%27s_method) algorithm.

    This method expects the following mandatory information:
    - `resistances`: EdgeValueInputs of the network, see [`Resistances`].
    - `current_src`: Current excitation of the network, see [`CurrentSources`].
    - `voltage_src`: Voltage excitation of the network, see [`VoltageSources`].

    Additionally, the iteration can be sped up with the following optional arguments:
    - `initial_edge_resistances`: If provided, these values will be used for the initial guess of the edge resistances
    (in further iterations, the Newton-Raphson algorithm calculates the edge resistances). If not provided,
    the initial guess is 1 for all edges. Providing a reasonable guess here (e.g. stemming from further knowledge
    of the system) can reduce the number of iterations and therefore speed up the solution process. The argument has
    no influence if the system can be solved directly.
    - `initial_edge_currents`: If provided, these values will be used for the initial guess of the edge currents
    (in further iterations, the Newton-Raphson algorithm calculates the edge currents). If not provided,
    the initial guess is 0 for all edges. See bullet point `initial_edge_resistances` for further information.
    - `jacobian`: See [`JacobianData`].

    Lastly, the Newton-Raphson algorithm requires some configuration parameters:
    - `config`: See [`SolverConfig`]. This struct provides reasonable defaults via its [`Default`] implementation.

    # Examples

    The following example shows how to solve a network with both current and voltage source and a resistance value
    which depends on the current going through it.

    ```
    use std::collections::HashMap;
    use approx; // Needed for result assertion
    use network_analysis::*;

    // Problem definition
    // =====================================================================

    /*
    This creates the following network with a current source at 0 and a voltage source at 3
     ┌─[1]─┬─[2]─┐
    [0]   [6]   [3]
     └─[5]─┴─[4]─┘
     */
    let mut edges: Vec<EdgeListEdge> = Vec::new();
    edges.push(EdgeListEdge::new(vec![5], vec![1], Type::Current));
    edges.push(EdgeListEdge::new(vec![0], vec![2, 6], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![1, 6], vec![3], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![2], vec![4], Type::Voltage));
    edges.push(EdgeListEdge::new(vec![3], vec![5, 6], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![4, 6], vec![0], Type::Resistance));
    edges.push(EdgeListEdge::new(vec![1, 2], vec![4, 5], Type::Resistance));
    let network = Network::from_edge_list_edges(&edges).expect("valid network");

    /*
    Resistance 6 is the square root of the current going through it plus an
    offset of 1. All other resistances are 1.
     */
    let resistances = Resistances::Function(&|mut args: FunctionArgs<'_>| {
        for (idx, val) in args.edge_value_and_type.iter_mut() {
            if idx == 6 {
                *val = 1.0 + args.edge_currents[6].abs().sqrt();
            } else {
                *val = 1.0;
            }
        }
    });

    let voltages = VoltageSources::Slice(&[0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0]);
    let currents = CurrentSources::Slice(&[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let config = SolverConfig::default();

    // Solving and analyzing the solution
    // =====================================================================

    let mut mesh_analysis = MeshAnalysis::new(&network);
    let solution_mesh = mesh_analysis.solve(resistances, currents, voltages, None, None, None, &config).expect("can be solved");

    // How many iterations were needed?
    assert_eq!(solution_mesh.iter_count(), 7);

    let mut nodal_analysis = NodalAnalysis::new(&network);
    let solution_nodal = nodal_analysis.solve(resistances, currents, voltages, None, None, None, &config).expect("can be solved");

    // How many iterations were needed?
    assert_eq!(solution_nodal.iter_count(), 7);

    for solution in [solution_mesh, solution_nodal].into_iter() {
        // Check the resistances
        let r: HashMap<_, _> = solution.resistances_and_indices().iter().collect();
        assert_eq!(r[&1], 1.0);
        approx::assert_abs_diff_eq!(r[&6], 1.532, epsilon = 1e-3);

        // Edge currents of 0, 1 and 5 are forced to be equal to the value of the current excitation
        assert_eq!(solution.currents()[0], 2.0);
        assert_eq!(solution.currents()[1], 2.0);
        assert_eq!(solution.currents()[5], 2.0);

        // Edge currents of 2, 3 and 4 are identical because of Kirchhoffs's law
        approx::assert_abs_diff_eq!(solution.currents()[2], 2.283, epsilon = 1e-3);
        approx::assert_abs_diff_eq!(solution.currents()[3], 2.283, epsilon = 1e-3);
        approx::assert_abs_diff_eq!(solution.currents()[4], 2.283, epsilon = 1e-3);

        // Kirchhoff's current law is fulfilled: Current through edge 6 is the difference
        // between the current through the edges 0, 1 and 5 and the current through the
        // edges 2, 3 and 4
        approx::assert_abs_diff_eq!(solution.currents()[6], -0.283, epsilon = 1e-3);
    }
    ```
    */
    fn solve<'a>(
        &'a mut self,
        resistances: Resistances,
        current_src: CurrentSources,
        voltage_src: VoltageSources,
        initial_edge_resistances: Option<&[f64]>,
        initial_edge_currents: Option<&[f64]>,
        jacobian: Option<&mut (dyn for<'b> FnMut(JacobianData<'b>) + 'a)>,
        config: &SolverConfig,
    ) -> Result<Solution<'a>, SolveError>;

    /**
    Returns the edge types.
     */
    fn edge_types(&self) -> &[Type];
}

/**
This trait provides the actual implementation of [`NetworkAnalysis::solve`].

The implementation relies on various private structs (to which the trait
needs to provide accessors). To avoid making all these structs public
and to minimize the interface of [`NetworkAnalysis`], this approach was chosen.
 */
pub(crate) trait NetworkAnalysisPriv: NetworkAnalysis {
    /**
    Calculates the edge currents and voltages from the network solution.
    */
    fn calculate_edge_currents_and_voltages(&mut self);

    /**
    Provides access to internals of the [`MeshAnalysis`] and [`NodalAnalysis`].
     */
    fn split_mut<'a>(
        &'a mut self,
    ) -> (
        &'a mut NetworkExcitation,
        &'a mut CoefficientMatrix,
        &'a DMatrix<f64>,
        &'a [Type],
        &'a mut Buffer,
    );

    fn solve<'a, const ANALYSIS_TYPE: u8>(
        &'a mut self,
        resistances: Resistances,
        current_src: CurrentSources,
        voltage_src: VoltageSources,
        initial_edge_resistances: Option<&[f64]>,
        initial_edge_currents: Option<&[f64]>,
        mut jacobian: Option<&mut (dyn for<'b> FnMut(JacobianData<'b>) + 'a)>,
        config: &SolverConfig,
    ) -> Result<Solution<'a>, SolveError> {
        // Limit scope of split_mut borrows.
        {
            let (network_excitation, _, _, edge_types, buf) = self.split_mut();

            // Check the input
            current_src.check(edge_types)?;
            voltage_src.check(edge_types)?;
            resistances.check(edge_types)?;

            // Reset the cached excitation and unknown buffers
            buf.x.iter_mut().for_each(|val| *val = 0.0);
            network_excitation.init_solve(&current_src, &voltage_src, edge_types);

            // Initialize the edge currents
            if let Some(slice) = initial_edge_currents {
                if slice.len() != edge_types.len() {
                    return Err(SolveError::SliceWrongLength {
                        slice_length: slice.len(),
                        edge_count: edge_types.len(),
                        slice_type: Type::Current,
                    });
                }
                buf.edge_currents.as_mut_slice().copy_from_slice(slice);
            } else {
                buf.edge_currents
                    .as_mut_slice()
                    .iter_mut()
                    .for_each(|val| *val = 0.0);
            }
        }

        self.calculate_edge_currents_and_voltages();

        // Limit scope of split_mut borrows.
        {
            let (network_excitation, coefficient_matrix, _, edge_types, buf) = self.split_mut();

            // The following procedure is structured according to figure 5.12 in:
            // Mathis, S.: Permanentmagneterregte Line-Start-Antriebe in Ferrittechnik,
            // PhD thesis, TU Kaiserslautern, Shaker-Verlag, 2019
            // "Anlaufrechnung"
            // Block 1: Initial coefficient matrix and excitation vector
            if resistances.is_const()
                && let Some(slice) = initial_edge_resistances
            {
                if slice.len() != edge_types.len() {
                    return Err(SolveError::SliceWrongLength {
                        slice_length: slice.len(),
                        edge_count: edge_types.len(),
                        slice_type: Type::Resistance,
                    });
                }
                buf.edge_resistances.as_mut_slice().copy_from_slice(slice);
            } else {
                resistances.calculate(FunctionArgs {
                    edge_value_and_type: EdgeValueAndTypeMut {
                        values: buf.edge_resistances.as_mut_slice(),
                        edge_types,
                        value_type: Type::Resistance,
                    },
                    edge_voltages: buf.edge_voltages.as_slice(),
                    edge_currents: buf.edge_currents.as_slice(),
                });
            }

            coefficient_matrix.calculate::<ANALYSIS_TYPE>(buf.edge_resistances.as_slice());

            network_excitation.calculate(
                buf.edge_currents.as_slice(),
                buf.edge_voltages.as_slice(),
                &current_src,
                &voltage_src,
                edge_types,
            );

            // Block 2: Solve for the mesh current vector
            // Store the values of b in x, since the direct solver replaces b in Ax = b with x.
            buf.x
                .as_mut_slice()
                .copy_from_slice(network_excitation.network().as_slice());

            // Perform a factorization into the buffer
            let mut buffer = buf.coefficients.take().expect("Buffer must not be empty");
            buffer
                .as_mut_slice()
                .copy_from_slice(coefficient_matrix.coefficient_matrix().as_slice());

            let lu = buffer.lu();
            lu.solve_mut(&mut buf.x);

            // Repopulate the buffer
            let _ = buf.coefficients.insert(lu.l_unpack());

            // Block 3: Store initial resistances values for the "-1" iteration
            buf.edge_resistances_prev
                .as_mut_slice()
                .copy_from_slice(buf.edge_resistances.as_slice());
        }

        // Initalize loop variables
        let mut iter_count: usize = 0;

        while iter_count < config.maxiter {
            // Block 4: Calculate edge current from mesh current
            self.calculate_edge_currents_and_voltages();

            // Limit scope of split_mut borrows.
            {
                let (network_excitation, coefficient_matrix, edge_to_system, edge_types, buf) =
                    self.split_mut();

                // Block 3: Generate initial values of the "-1" iteration
                if iter_count == 0 {
                    buf.edge_currents_prev
                        .as_mut_slice()
                        .copy_from_slice(buf.edge_currents.as_slice());
                    buf.edge_currents_prev
                        .iter_mut()
                        .for_each(|x| *x *= config.prev_value_variation);
                }

                // Block 5,6,7: Calculate the new coefficient matrices and excitations
                let coefficient_matrix = if !resistances.is_const() {
                    resistances.calculate(FunctionArgs {
                        edge_value_and_type: EdgeValueAndTypeMut {
                            values: buf.edge_resistances.as_mut_slice(),
                            edge_types,
                            value_type: Type::Resistance,
                        },
                        edge_voltages: buf.edge_voltages.as_slice(),
                        edge_currents: buf.edge_currents.as_slice(),
                    });
                    coefficient_matrix.calculate::<ANALYSIS_TYPE>(buf.edge_resistances.as_slice())
                } else {
                    coefficient_matrix.coefficient_matrix()
                };

                network_excitation.calculate(
                    buf.edge_currents.as_slice(),
                    buf.edge_voltages.as_slice(),
                    &current_src,
                    &voltage_src,
                    edge_types,
                );

                // Calculate the residual error f(x) = A*x - b (iteration finishes if this value is zero)
                // ======================================================================================

                // Calculate = A*x
                buf.f_x.gemm(1.0, coefficient_matrix, &buf.x, 0.0);

                // Calculate (A*x) - b
                buf.f_x
                    .iter_mut()
                    .zip(network_excitation.network().iter())
                    .for_each(|(f_x, b)| {
                        *f_x = *f_x - *b;
                    });

                // Check the maximum change of resistances, elementwise. If all changes
                // are below a threshold defined by ϵ, the loop is stopped.
                if config.is_root(buf.f_x.as_slice()) {
                    break;
                }

                // Block 11: Update the iteration counter
                iter_count += 1;

                if iter_count >= config.maxiter {
                    return Err(SolveError::MaxIterReached(config.maxiter));
                }

                // Block 9: Calculate the Jacobian
                match jacobian.as_mut() {
                    Some(jac_fn) => {
                        jac_fn(JacobianData {
                            jacobian: &mut buf.jac,
                            coefficient_matrix,
                            unknowns: &buf.x,
                            edge_types: edge_types,
                            edge_to_system,
                            excitation: network_excitation.network(),
                            edge_resistances: &buf.edge_resistances,
                            edge_resistances_prev: &buf.edge_resistances_prev,
                            edge_currents: &buf.edge_currents,
                            edge_currents_prev: &buf.edge_currents_prev,
                            edge_voltages: &buf.edge_voltages,
                            edge_voltages_prev: &buf.edge_voltages_prev,
                        });
                    }
                    None => {
                        // Define the Jacobian function as f(x) = Ax - b, where A is the system matrix,
                        // x is the x vector and b is the b vector.
                        let jac_function = |f_x: &mut [f64], x: &[f64]| {
                            buf.x_for_jac.as_mut_slice().copy_from_slice(x);
                            buf.f_x_for_jac
                                .gemm(1.0, coefficient_matrix, &buf.x_for_jac, 0.0);

                            f_x.iter_mut()
                                .zip(buf.f_x_for_jac.iter())
                                .zip(network_excitation.network().iter())
                                .for_each(|((y, f_x_for_jac), b)| {
                                    *y = *f_x_for_jac - *b;
                                });
                        };
                        central_jacobian(
                            buf.jac.as_mut_slice(),
                            buf.x.as_slice(),
                            config.finite_difference_stepwidth,
                            buf.jac_x.as_mut_slice(),
                            buf.jac_f_x.as_mut_slice(),
                            jac_function,
                        );
                    }
                }

                // Check if the Jacobian contains a NaN-element. If this is the case, the solver failed.
                if buf.jac.iter().any(|x| x.is_nan()) {
                    return Err(SolveError::SingularJacobian);
                }

                // Introduce the dampening factor for the Jacobian
                if config.dampening != 1.0 {
                    buf.jac.scale_mut(config.dampening);
                }

                // Calculate f(x) / f'(x) into self.f_x
                let mut buffer = buf.coefficients.take().expect("Buffer must not be empty");
                buffer.as_mut_slice().copy_from_slice(buf.jac.as_slice());

                let lu = buffer.lu();
                lu.solve_mut(&mut buf.f_x);

                // Repopulate the buffer
                let _ = buf.coefficients.insert(lu.l_unpack());

                // Update x_n+1 = x_n - f(x) / f'(x)
                buf.x
                    .as_mut_slice()
                    .iter_mut()
                    .zip(buf.f_x.iter())
                    .for_each(|(x, f_x)| {
                        *x = *x - *f_x;
                    });

                // Store the resistance / current / voltage values of the previous iteration
                buf.update_prev();
            }
        }

        let (_, _, _, edge_types, buf) = self.split_mut();
        return Ok(Solution::new(buf, edge_types, iter_count));
    }
}

// ==========================================================================

/**
Solver parameters used in the [`NetworkAnalysis::solve`] function.

The default values given for each field are used in the implementation of the [`Default`] trait.
*/
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SolverConfig {
    /**
    This value is used to create a "zeroth" iteration value which is necessary to start the first iteration. This is a heuristic value
    which can be varied if the solver does not converge. Default is 0.9.
     */
    pub prev_value_variation: f64,
    /**
    The Newton-Raphson algorithmus used in the implementations of [`NetworkAnalysis`] uses the [`ulps_eq`] from the [approx] crate in order
    to determine whether the root has been found with sufficient precision (i.e., `f(x_0) == 0` is approximately fulfilled). [`ulps_eq`] uses both an absolute
    difference check (using the `epsilon` field) and an ULPs (using `max_ulps` field) comparison. For the former check,  is used.
    Default is the square root of the machine precision (`std::f64::EPSILON.sqrt()`).
     */
    pub epsilon: f64,
    ///  See `epsilon`. Default is 4.
    pub max_ulps: u32,
    /// Maximum number of iterations of the underlying Newton-Raphson solver. When `maxiter` is exceeded, the solver terminates with an error. Default is 200.
    pub maxiter: usize,
    /**
    If no custom Jacobian function is given, the Jacobian is calculated as a finite difference approximation (see \[1\])
    This field `finite_difference_stepwidth` corresponds to the stepwidth `h`.
    Default is the square root of the machine precision (`std::f64::EPSILON.sqrt()`).

    1) Pauletti, Ruy & Almeida Neto, Edgard: A finite-difference approximation to Newton´s Method Jacobian Matrices. IABSE-IASS Symposium London 2011).
     */
    pub finite_difference_stepwidth: f64,
    /**
    As described in \[2\], the dampening factor tau can be used to increase the convergence
    probability at the cost of higher iteration time. Default is 1.0 (no dampening).

    2) Okawa, Hirotada et al.: The W4 method: A new multi-dimensional root-finding scheme for nonlinear systems of equations. Applied Numerical Mathematics 183 (2023) 157-172.
     */
    pub dampening: f64,
}

impl SolverConfig {
    /**
    Asserts that the input vector is approximately zero.
    */
    #[inline]
    pub(crate) fn is_root(&self, x: &[f64]) -> bool {
        return x.iter().all(|x| {
            ulps_eq!(
                x.abs(),
                0.0,
                epsilon = self.epsilon,
                max_ulps = self.max_ulps
            )
        });
    }
}

impl Default for SolverConfig {
    /**
    See the docstring of [`SolverConfig`] for the default parameters values.
     */
    fn default() -> Self {
        Self {
            prev_value_variation: 0.9,
            epsilon: std::f64::EPSILON.sqrt(),
            max_ulps: 4,
            maxiter: 200,
            finite_difference_stepwidth: std::f64::EPSILON.sqrt(),
            dampening: 1.0,
        }
    }
}

pub type Resistances<'a> = EdgeValueInputs<'a, RESISTANCE>;
pub type CurrentSources<'a> = EdgeValueInputs<'a, CURRENT>;
pub type VoltageSources<'a> = EdgeValueInputs<'a, VOLTAGE>;

/// Marker trait which restricts the constant value for [`EdgeValueInputs`] to 0 (= [`Resistances`]),
/// 1 (= [`CurrentSources`]) and 2 (= [`CurrentSources`]). This trait needs to have public visibility,
/// but it doesn't do anything other than the aforementioned restriction and is therefore sealed.
pub trait ValidTypeValues: private::Sealed {}

mod private {
    pub trait Sealed {}
}

// Restruct possible TYPE values of EdgeValueInputs.
impl private::Sealed for Resistances<'_> {}
impl private::Sealed for CurrentSources<'_> {}
impl private::Sealed for VoltageSources<'_> {}
impl ValidTypeValues for Resistances<'_> {}
impl ValidTypeValues for CurrentSources<'_> {}
impl ValidTypeValues for VoltageSources<'_> {}

/**
Enum defining the edge input values for current / voltage excitations or resistances as either constants or via functions.

Resistance values must always be strictly positive (larger than zero).
The generic `TYPE` is:
- 0 for resistances (alias: [`Resistances`])
- 1 for current sources (alias: [`CurrentSources`])
- 2 for voltage sources (alias: [`VoltageSources`])
 */
#[derive(Clone, Copy)]
pub enum EdgeValueInputs<'a, const TYPE: u8>
where
    EdgeValueInputs<'a, TYPE>: ValidTypeValues,
{
    /**
    If all current / voltage excitations or resistances are constant, it can be convenient to provide their values as a slice.
    This slice has to fulfill the following constraints:
    - Its length must be equal to the number of edges.
    - If an edge has another type, its corresponding entry must be zero.
    - If an edge is a resistance, its corresponding slice entry must be larger than zero.
    These conditions are checked at the start of a call to [`NetworkAnalysis::solve`] and a
    [`SolveError`] is returned if they're not fulfilled.
     */
    Slice(&'a [f64]),
    /**
    This is an alternative to [`EdgeValueInputs::Slice`] where the values are provided directly
    together with their indices as a tuple `(index, value)` - useful if the network does not contain many current / voltage excitations or resistances.

    For example, these two `Resistances` lead in the same resultat when used in a `solve` call:
    ```
    use network_analysis::Resistances;

    let res1 = Resistances::Slice(&[0.0, 0.0, 1.0, 2.0, 0.0]);
    let res1 = Resistances::IdxAndVals(&[(2, 1.0f64), (3, 2.0f64)]);
    ```

    The indices have to fulfill the following constraints:
    - They must be within bounds (i.e. smaller than the number of edges)-
    - They must not contain duplicates.
    - They must not index edges of a different type.
    Additionally, the values must always be larger than zero
    These conditions are checked at the start of a call to [`NetworkAnalysis::solve`] and a
    [`SolveError`] is returned if they're not fulfilled.
     */
    IdxAndVals(&'a [(usize, f64)]),
    /**
    This variant can be used to represent variable resistances - for example, resistances whose
    value depends on the current going through them. It wraps a user-provided function pointer
    which takes a single argument [`FunctionArgs`]. This argument contains the edge currents
    and voltages of the current iteration and a [`EdgeValueAndTypeMut`] wrapper over a mutable
    edge resistance slice. See [`EdgeValueAndTypeMut::iter_mut`] for an example.
    */
    Function(&'a dyn Fn(FunctionArgs<'_>)),
    /**
    If no edges of the specified type exist within the network, this variant can be used.
     */
    None,
}

impl<'a, const TYPE: u8> EdgeValueInputs<'a, TYPE>
where
    EdgeValueInputs<'a, TYPE>: ValidTypeValues,
{
    /**
    Check if `self` is valid according to the constraints outlined in the docstrings of the variants.
     */
    pub fn check(&self, edge_types: &[Type]) -> Result<(), SolveError> {
        let type_of_self = match TYPE {
            0 => Type::Resistance,
            1 => Type::Current,
            2 => Type::Voltage,
            _ => unreachable!(),
        };

        let edge_count = edge_types.len();
        match self {
            EdgeValueInputs::Slice(slice) => {
                if slice.len() != edge_count {
                    return Err(SolveError::SliceWrongLength {
                        slice_type: type_of_self,
                        slice_length: slice.len(),
                        edge_count,
                    });
                }
                for (idx, (val, edge_type)) in slice
                    .iter()
                    .cloned()
                    .zip(edge_types.iter().cloned())
                    .enumerate()
                {
                    if edge_type == type_of_self {
                        if type_of_self == Type::Resistance {
                            if val <= 0.0 {
                                return Err(SolveError::NonPositiveResistance { val, idx });
                            }
                        }
                    } else {
                        if val != 0.0 && edge_type != type_of_self {
                            return Err(SolveError::NonzeroForMismatchedEdgeType {
                                val,
                                idx,
                                edge_type_at_idx: edge_type,
                                edge_type_check: type_of_self,
                            });
                        }
                    }
                }
            }
            EdgeValueInputs::IdxAndVals(items) => {
                for (idx, val) in items.iter().cloned() {
                    if val < 0.0 {
                        return Err(SolveError::NonPositiveResistance { val, idx });
                    }

                    let quantity = edge_types[idx];
                    if val != 0.0 && quantity != type_of_self {
                        return Err(SolveError::NonzeroForMismatchedEdgeType {
                            val,
                            idx,
                            edge_type_at_idx: quantity,
                            edge_type_check: type_of_self,
                        });
                    }
                    if idx >= edge_count {
                        return Err(SolveError::OutOfBoundsIdx {
                            idx_and_val_type: type_of_self,
                            idx,
                            edge_count,
                        });
                    }
                }
            }
            _ => (),
        }
        return Ok(());
    }

    /**
    Populate the edge values given in `args` (field `edge_value_and_type.values`) using `self`.

    # Example
    ```
    use network_analysis::{Type, EdgeValueAndTypeMut, FunctionArgs, Resistances};

    fn resistance_calc(mut input: FunctionArgs<'_>) {
        for (idx, resistance) in input.edge_value_and_type.iter_mut() {
            // SAFETY: When using `FunctionArgs`, the indices returned by the iterator are guaranteed to be in bounds of the edge slices.
            let edge_current = unsafe{input.edge_currents.get_unchecked(idx)};
            *resistance = 1.0 + edge_current.powi(2);
        }
    }

    let resistances = &mut [0.0, 0.0, 0.0, 0.0];
    let edge_types = &[Type::Resistance, Type::Voltage, Type::Resistance, Type::Resistance];
    let edge_value_and_type = EdgeValueAndTypeMut {
        values: resistances,
        edge_types,
        value_type: Type::Resistance // Filter for resistances
    };

    // This struct is provided `NetworkAnalysis::solve` when using it for analyzing a network
    let input = FunctionArgs {
        edge_value_and_type,
        edge_currents: &[1.0, 2.0, 3.0, 4.0], // Arbitrary numbers
        edge_voltages: &[1.0, 2.0, 3.0, 4.0], // Arbitrary numbers
    };

    let resistance_enum = Resistances::Function(&resistance_calc);

    resistance_enum.calculate(input);

    // Check the populated values
    assert_eq!(resistances[0], 2.0);
    assert_eq!(resistances[1], 0.0); // Filtered out
    assert_eq!(resistances[2], 10.0);
    assert_eq!(resistances[3], 17.0);
    ```

     */
    pub fn calculate(&self, args: FunctionArgs<'_>) {
        match self {
            EdgeValueInputs::Slice(slice) => {
                for (dst, src) in args.edge_value_and_type.values.iter_mut().zip(slice.iter()) {
                    /*
                    If the slice value is zero, the old entry should be kept.
                    This is checked here branchless.
                    */
                    *dst = *src + *dst * ((*src == 0.0) as u8 as f64);
                }
            }
            EdgeValueInputs::IdxAndVals(items) => {
                for (idx, val) in items.iter() {
                    // SAFETY: At the beginning of the solve call, it was asserted that all indices are valid
                    let res_ref =
                        unsafe { args.edge_value_and_type.values.get_unchecked_mut(*idx) };
                    *res_ref = *val;
                }
            }
            EdgeValueInputs::Function(fun) => {
                fun(args);
            }
            EdgeValueInputs::None => (),
        }
    }

    /**
    Returns true, if the excitation is constant (true for all enum variants except [`EdgeValueInputs::Function`]).

    # Example
    ```
    use network_analysis::{FunctionArgs, Resistances, CurrentSources, VoltageSources};

    fn resistance_calc(mut input: FunctionArgs<'_>) {
        for (idx, resistance) in input.edge_value_and_type.iter_mut() {
            // SAFETY: When using `FunctionArgs`, the indices returned by the iterator are guaranteed to be in bounds of the edge slices.
            let edge_current = unsafe{input.edge_currents.get_unchecked(idx)};
            *resistance = 1.0 + edge_current.powi(2);
        }
    }

    assert!(VoltageSources::Slice(&[]).is_const());
    assert!(CurrentSources::IdxAndVals(&[]).is_const());
    assert!(!Resistances::Function(&resistance_calc).is_const());
    ```
     */
    pub fn is_const(&self) -> bool {
        match self {
            EdgeValueInputs::Function(_) => false,
            _ => true,
        }
    }
}

/**
A struct containing information about the nonlinear equation system (both its general structure and the concrete values of its component).
It is provided as an input to a user-supplied Jacobian calculation function,

The network solver tries to solve the general matrix equation `A * x = b`, with `A` being the coefficient matrix,
`x` being the unknowns (the mesh currents for [`MeshAnalysis`](`crate::MeshAnalysis`) or the node potential for [`NodalAnalysis`](`crate::NodalAnalysis`)
and `b` being the excitations (current and / or voltage sources).

This is equivalent to searching the root of `A * x - b = 0`, hence the Newton-Raphson root finding algorithm can be applied. This requires calculating
the Jacobian, where every equation of the system is derived by every value of `x` (see <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>).
By default, the network solver uses a finite differences approximation of the Jacobian, but providing an analytical calculation method function return
better results with regards to the number of iterations needed.

The following example shows how to write a custom Jacobian calculation function.
```
use approx; // Needed for result assertion
use network_analysis::*;

// Problem definition
// =====================================================================

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

/*
All resistances are the square of the current going through them + 1
The resulting equation system is therefore nonlinear and requires iterations to be solved.
*/
let resistances = EdgeValueInputs::Function(&|mut args: FunctionArgs<'_>| {
    for (idx, val) in args.edge_value_and_type.iter_mut() {
         *val = 1.0 + args.edge_currents[idx].powi(2);
    }
});

let voltages = EdgeValueInputs::Slice(&[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
let config = SolverConfig::default();

/**
The derivative of `b` with respect to `x` is zero (voltage sources are constant).
The derivative `d(A * x)/dx` can be rewritten as `dA/dx * x + A * dx/dx` (product rule).
`dx/dx` is the unity vector, hence the equation simplifies to `dA/dx * x + A`.

To calculate `dA/dx`, one needs to understand that the elements (sum of resistances in this case) of `A` are a
function of the individual edge currents, not of `x` (`x` is a sum of multiple edge currents).
Therefore, the following formula results for deriving any matrix element by x:
`dA[row, col]/dx[row] = d(edge_current[0])/dx[row] * d(edge_resistance[0])/d(edge_current[0]) + d(edge_current[1])/dx[row] * d(edge_resistance[1])/d(edge_current[1]) + ...`
The value of `d(edge_current[edge])/dx[row]` is equal to `-data.edge_to_system[(row, edge)]` (which describes the coupling between meshes and edges).
*/
fn calc_jacobian(data: JacobianData<'_>) {
    let ncols = data.jacobian.ncols();
    let nrows = data.jacobian.nrows();
    let nedges = data.edge_types.len();

    // DMatrix is column-major
    for col in 0..ncols {
        for row in 0..nrows {
            let mut sum = data.coefficient_matrix[(row, col)]; // this is A[row, col] from the function docstring
            for (edge, edge_type) in data.edge_types.iter().enumerate() {
                // Filter out excitation edges
                if Type::Resistance == *edge_type {
                    // Read out the negative coupling from the system matrix
                    let coupling = -data.edge_to_system[(row, edge)];

                    // Derivative of `edge_currents[idx].powi(2)` is:
                    let derivative = coupling * 2.0 * data.edge_currents[edge];

                    // Add to entry
                    sum += derivative;
                }
            }
            data.jacobian[(row, col)] = sum;
        }
    }
}

// Solving without finite difference approximation
let sol = mesh_analysis.solve(resistances, EdgeValueInputs::None, voltages, None, None, None, &config).expect("can be solved");
let edge_currents_fd = sol.currents().to_vec();
assert_eq!(sol.iter_count(), 119);

// Solving with analytical Jacobian
let sol = mesh_analysis.solve(resistances, EdgeValueInputs::None, voltages, None, None, Some(&mut calc_jacobian), &config).expect("can be solved");
let edge_currents_an = sol.currents().to_vec();
assert_eq!(sol.iter_count(), 34);

for (curr_fd, curr_an) in edge_currents_fd.iter().cloned().zip(edge_currents_an.iter().cloned()) {
    approx::assert_abs_diff_eq!(curr_fd, curr_an, epsilon = 1e-6);
}
```
 */
pub struct JacobianData<'a> {
    /// Jacobian buffer. The user supplied-function is expected to populate this matrix.
    pub jacobian: &'a mut DMatrix<f64>,
    /// The `A` in `A * x = b`
    pub coefficient_matrix: &'a DMatrix<f64>,
    /// The `x` in `A * x = b`
    pub unknowns: &'a DVector<f64>,
    /// The `b` in `A * x = b`
    pub excitation: &'a DVector<f64>,
    /**
    This is the system structure matrix ([`MeshAnalysis::edge_to_mesh`](`crate::MeshAnalysis::edge_to_mesh`) or
    [`NodalAnalysis::edge_to_node`](`crate::NodalAnalysis::edge_to_node`)). It contains all information regarding
    the structure of the equation system, please see the documentation of the respective methods for more.
     */
    pub edge_to_system: &'a DMatrix<f64>,
    /// Edge types of the network.
    pub edge_types: &'a [Type],
    /// Edge resistance calculated in the current iteration
    pub edge_resistances: &'a DVector<f64>,
    /// Edge resistance calculated in the last iteration.
    pub edge_resistances_prev: &'a DVector<f64>,
    /// Current through the edges calculated in the current iteration
    pub edge_currents: &'a DVector<f64>,
    /// Current through the edges calculated in the last iteration
    pub edge_currents_prev: &'a DVector<f64>,
    /// Voltage drop over the edges calculated in the current iteration.
    pub edge_voltages: &'a DVector<f64>,
    /// Voltage drop over the edges calculated in the last iteration.
    pub edge_voltages_prev: &'a DVector<f64>,
}

/**
Struct to calculate the coefficient / system matrix (the `A` in `A * x = b`).
 */
#[derive(Debug, Clone)]
pub(crate) struct CoefficientMatrix {
    coefficient_matrix: DMatrix<f64>,
    edge_to_network_resistance: DMatrix<Vec<f64>>,
}

impl CoefficientMatrix {
    pub(crate) fn new(
        coefficient_matrix: DMatrix<f64>,
        edge_to_network_resistance: DMatrix<Vec<f64>>,
    ) -> Self {
        assert!(coefficient_matrix.ncols() >= edge_to_network_resistance.ncols());
        assert!(coefficient_matrix.nrows() >= edge_to_network_resistance.nrows());
        return Self {
            coefficient_matrix,
            edge_to_network_resistance,
        };
    }

    /**
    Update the internally cached network coefficient matrix (`self.coefficient_matrix`) and return a reference to it.

    Returns `(edge_resistances, coefficient_matrix)`.
     */
    pub(crate) fn calculate<const ANALYSIS_TYPE: u8>(
        &mut self,
        edge_resistances: &[f64],
    ) -> &DMatrix<f64> {
        for col in 0..self.edge_to_network_resistance.ncols() {
            for row in 0..self.edge_to_network_resistance.nrows() {
                // SAFETY: The validity of the indices has been varified in the constructor
                let vec = unsafe { self.edge_to_network_resistance.get_unchecked((row, col)) };
                if vec.len() > 0 {
                    let coeff = unsafe { self.coefficient_matrix.get_unchecked_mut((row, col)) };
                    *coeff = vec
                        .as_slice()
                        .par_iter()
                        .cloned()
                        .zip(edge_resistances.par_iter().cloned())
                        .map(|(coupling, edge)| {
                            if ANALYSIS_TYPE == MESH_ANALYSIS {
                                coupling * edge
                            } else if ANALYSIS_TYPE == NODAL_ANALYSIS {
                                // Branchless algorithm to avoid dividing by zero
                                let corrected_edge = (coupling == 0.0) as u8 as f64 + edge;
                                coupling / corrected_edge
                            } else {
                                unreachable!("function is only valid for mesh or nodal analysis")
                            }
                        })
                        .sum();
                }
            }
        }
        return &self.coefficient_matrix;
    }

    pub(crate) fn edge_to_network_resistance(&self) -> &DMatrix<Vec<f64>> {
        return &self.edge_to_network_resistance;
    }

    pub(crate) fn coefficient_matrix(&self) -> &DMatrix<f64> {
        return &self.coefficient_matrix;
    }
}

/**
Struct to calculate the edge-wise excitation and converting it into a network excitation (the `b` in `A * x = b`).
 */
#[derive(Debug, Clone)]
pub(crate) struct NetworkExcitation {
    constant: bool,
    network: DVector<f64>,
    edges: DVector<f64>,
    conversion: DMatrix<f64>, // Convert from edge to network excitation
}

impl NetworkExcitation {
    pub(crate) fn new(conversion: DMatrix<f64>) -> Self {
        let edges = DVector::repeat(conversion.ncols(), 0.0);
        let network = DVector::repeat(conversion.nrows(), 0.0);
        return Self {
            constant: false,
            network,
            edges,
            conversion,
        };
    }

    /**
    Update the internally cached network excitation (`self.network`) and return a reference to it.
     */
    pub(crate) fn calculate(
        &mut self,
        edge_currents: &[f64],
        edge_voltages: &[f64],
        current_src: &CurrentSources,
        voltage_src: &VoltageSources,
        edge_types: &[Type],
    ) -> &DVector<f64> {
        if self.constant {
            return &self.network;
        }

        current_src.calculate(FunctionArgs {
            edge_value_and_type: EdgeValueAndTypeMut {
                values: self.edges.as_mut_slice(),
                edge_types,
                value_type: Type::Current,
            },
            edge_voltages,
            edge_currents,
        });
        voltage_src.calculate(FunctionArgs {
            edge_value_and_type: EdgeValueAndTypeMut {
                values: self.edges.as_mut_slice(),
                edge_types,
                value_type: Type::Voltage,
            },
            edge_voltages,
            edge_currents,
        });

        // Invert voltage sources
        for (val, edge_type) in self.edges.iter_mut().zip(edge_types.iter()) {
            if *edge_type == Type::Voltage {
                *val = -*val;
            }
        }

        // Transform the edge excitation into the network excitation
        self.network.gemm(1.0, &self.conversion, &self.edges, 0.0);

        return &self.network;
    }

    /**
    This function needs to run at the begin of each solve call.
     */
    pub(crate) fn init_solve(
        &mut self,
        current_src: &CurrentSources,
        voltage_src: &VoltageSources,
        edge_types: &[Type],
    ) {
        // Reset the vectors from previous solve calls
        self.edges.iter_mut().for_each(|elem| *elem = 0.0);
        self.network.iter_mut().for_each(|elem| *elem = 0.0);

        // If the excitation is constant, precalculate the network excitation and cache it.
        self.constant = current_src.is_const() && voltage_src.is_const();
        if self.constant {
            // The edge voltages and currents are not needed when the excitation is constant
            current_src.calculate(FunctionArgs {
                edge_value_and_type: EdgeValueAndTypeMut {
                    values: self.edges.as_mut_slice(),
                    edge_types,
                    value_type: Type::Current,
                },
                edge_voltages: &[],
                edge_currents: &[],
            });
            voltage_src.calculate(FunctionArgs {
                edge_value_and_type: EdgeValueAndTypeMut {
                    values: self.edges.as_mut_slice(),
                    edge_types,
                    value_type: Type::Voltage,
                },
                edge_voltages: &[],
                edge_currents: &[],
            });

            for (val, edge_type) in self.edges.iter_mut().zip(edge_types.iter()) {
                if *edge_type == Type::Voltage {
                    *val = -*val;
                }
            }

            // Transform the edge excitation into the mesh excitation
            self.network.gemm(1.0, &self.conversion, &self.edges, 0.0);
        }
    }

    /**
    Returns the cached network excitation vector (the `b` in `A * x = b`).
     */
    pub(crate) fn network(&self) -> &DVector<f64> {
        return &self.network;
    }

    /**
    Returns the edgewise excitation (both current and voltage excitation).
     */
    pub(crate) fn edges(&self) -> &DVector<f64> {
        return &self.edges;
    }
}

/**
Buffers needed for both mesh analysis and nodal analysis.
 */
#[derive(Debug, Clone)]
pub(crate) struct Buffer {
    pub(crate) edge_resistances: DVector<f64>,
    pub(crate) edge_resistances_prev: DVector<f64>,

    pub(crate) edge_currents: DVector<f64>,
    pub(crate) edge_currents_prev: DVector<f64>,

    pub(crate) edge_voltages: DVector<f64>,
    pub(crate) edge_voltages_prev: DVector<f64>,

    // Buffers for network equation
    pub(crate) x: DVector<f64>,
    pub(crate) f_x: DVector<f64>,
    pub(crate) x_for_jac: DVector<f64>,
    pub(crate) f_x_for_jac: DVector<f64>,
    pub(crate) jac: DMatrix<f64>,
    pub(crate) coefficients: Option<DMatrix<f64>>,

    // Buffers for the calcution of the jacobian
    pub(crate) jac_x: Vec<f64>,
    pub(crate) jac_f_x: Vec<f64>,
}

impl Buffer {
    pub(crate) fn new(edge_count: usize, network_equations: usize) -> Self {
        return Self {
            edge_resistances: DVector::repeat(edge_count, 0.0),
            edge_resistances_prev: DVector::repeat(edge_count, 0.0),
            edge_currents: DVector::repeat(edge_count, 0.0),
            edge_currents_prev: DVector::repeat(edge_count, 0.0),
            edge_voltages: DVector::repeat(edge_count, 0.0),
            edge_voltages_prev: DVector::repeat(edge_count, 0.0),
            x: DVector::repeat(network_equations, 0.0),
            f_x: DVector::repeat(network_equations, 0.0),
            x_for_jac: DVector::repeat(network_equations, 0.0),
            f_x_for_jac: DVector::repeat(network_equations, 0.0),
            jac: DMatrix::repeat(network_equations, network_equations, 0.0),
            coefficients: Some(DMatrix::repeat(network_equations, network_equations, 0.0)),
            jac_x: vec![0.0; network_equations],
            jac_f_x: vec![0.0; network_equations],
        };
    }

    /**
    Copy the buffers of edge resistance, edge currents and edge voltages into the
    `_prev` buffers.
     */
    pub(crate) fn update_prev(&mut self) {
        self.edge_resistances_prev
            .as_mut_slice()
            .copy_from_slice(self.edge_resistances.as_slice());
        self.edge_currents_prev
            .as_mut_slice()
            .copy_from_slice(self.edge_currents.as_slice());
        self.edge_voltages_prev
            .as_mut_slice()
            .copy_from_slice(self.edge_voltages.as_slice());
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_network_excitation() {
        fn test_network_excitation_inner(
            first_curr: CurrentSources,
            first_volt: VoltageSources,
            second_curr: CurrentSources,
            second_volt: VoltageSources,
        ) {
            let mut mat = DMatrix::repeat(3, 7, 0.0);
            mat[(0, 5)] = 1.0;
            mat[(1, 0)] = 1.0;
            let edge_types = &[
                Type::Voltage,
                Type::Resistance,
                Type::Resistance,
                Type::Resistance,
                Type::Resistance,
                Type::Current,
                Type::Resistance,
            ];
            let mut exc_calculator = NetworkExcitation::new(mat);
            {
                exc_calculator.calculate(&[], &[], &first_curr, &first_volt, edge_types);

                let edge_exc: Vec<f64> =
                    exc_calculator.edges().as_slice().iter().cloned().collect();
                assert_eq!(edge_exc, vec![-1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0]);

                let net_exc: Vec<f64> = exc_calculator
                    .network()
                    .as_slice()
                    .iter()
                    .cloned()
                    .collect();
                assert_eq!(net_exc, vec![5.0, -1.0, 0.0]);
            }

            {
                exc_calculator.calculate(&[], &[], &second_curr, &second_volt, edge_types);

                let edge_exc: Vec<f64> =
                    exc_calculator.edges().as_slice().iter().cloned().collect();
                assert_eq!(edge_exc, vec![3.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0]);

                let net_exc: Vec<f64> = exc_calculator
                    .network()
                    .as_slice()
                    .iter()
                    .cloned()
                    .collect();
                assert_eq!(net_exc, vec![-2.0, 3.0, 0.0]);
            }
        }

        test_network_excitation_inner(
            CurrentSources::Slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0]),
            EdgeValueInputs::Slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            CurrentSources::Slice(&[0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0]),
            EdgeValueInputs::Slice(&[-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        );

        test_network_excitation_inner(
            CurrentSources::IdxAndVals(&[(5, 5.0)]),
            EdgeValueInputs::IdxAndVals(&[(0, 1.0)]),
            CurrentSources::IdxAndVals(&[(5, -2.0)]),
            EdgeValueInputs::IdxAndVals(&[(0, -3.0)]),
        );

        let first_curr = |mut args: FunctionArgs| {
            let (_, val) = args.edge_value_and_type.iter_mut().next().unwrap();
            *val = 5.0;
        };
        let first_volt = |mut args: FunctionArgs| {
            let (_, val) = args.edge_value_and_type.iter_mut().next().unwrap();
            *val = 1.0;
        };
        let second_curr = |mut args: FunctionArgs| {
            let (_, val) = args.edge_value_and_type.iter_mut().next().unwrap();
            *val = -2.0;
        };
        let second_volt = |mut args: FunctionArgs| {
            let (_, val) = args.edge_value_and_type.iter_mut().next().unwrap();
            *val = -3.0;
        };
        test_network_excitation_inner(
            CurrentSources::Function(&first_curr),
            VoltageSources::Function(&first_volt),
            CurrentSources::Function(&second_curr),
            VoltageSources::Function(&second_volt),
        );
    }

    #[test]
    fn test_coefficient_matrix() {
        let mut coeff_mat = DMatrix::repeat(3, 3, 0.0);
        coeff_mat[(0, 0)] = 1.0;

        let mut matrix = DMatrix::repeat(3, 3, Vec::new());
        matrix[(1, 1)] = vec![0.0, 1.0, 1.0, 1.0];
        matrix[(2, 2)] = vec![1.0, 1.0, 0.0, 0.0];
        matrix[(1, 2)] = vec![-1.0, 0.0, 0.0, 0.0];
        matrix[(2, 1)] = matrix[(1, 2)].clone();

        let mut coefficient_matrix = CoefficientMatrix::new(coeff_mat, matrix);
        let res = coefficient_matrix.calculate::<MESH_ANALYSIS>(&[1.5, 0.5, 2.7, 0.1]);

        approx::assert_abs_diff_eq!(res[(0, 0)], 1.0, epsilon = 1e-3);
        approx::assert_abs_diff_eq!(res[(1, 0)], 0.0, epsilon = 1e-3);
        approx::assert_abs_diff_eq!(res[(2, 0)], 0.0, epsilon = 1e-3);
        approx::assert_abs_diff_eq!(res[(0, 1)], 0.0, epsilon = 1e-3);
        approx::assert_abs_diff_eq!(res[(1, 1)], 3.3, epsilon = 1e-3);
        approx::assert_abs_diff_eq!(res[(2, 1)], -1.5, epsilon = 1e-3);
        approx::assert_abs_diff_eq!(res[(0, 2)], 0.0, epsilon = 1e-3);
        approx::assert_abs_diff_eq!(res[(1, 2)], -1.5, epsilon = 1e-3);
        approx::assert_abs_diff_eq!(res[(2, 2)], 2.0, epsilon = 1e-3);
    }
}
