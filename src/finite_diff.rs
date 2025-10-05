/*!
This module contains a finite-difference approximation of the Jacobian matrix.
 */

/**
Returns an in-place approximation of the Jacobian `J = d function(x) / dx`.

The slice `jac` represents a flattened column-major Jacobian matrix.
For example, the slice [1 2 3 4] of a two-column Jacobian represents the matrix:
[1 3]
[2 4]
The number of columns and rows are derived from the inputs `jac` and `x`:
`ncols = x.len()`
`nrows = jac.len() / ncols`

This function is based on \[1\].

# Literature

1) Pauletti, Ruy & Almeida Neto, Edgard. (2018). A finite-difference approximation to Newton´s Method Jacobian Matrices.
IABSE-IASS Symposium London 2011
```
*/
pub(crate) fn central_jacobian(
    jac: &mut [f64],
    x: &[f64],
    h: f64,
    buf_x: &mut [f64],
    buf_fx: &mut [f64],
    mut f: impl FnMut(&mut [f64], &[f64]),
) {
    let ncols = x.len();
    let nrows = jac.len() / ncols;
    let reciprocal = 0.5 / h;

    assert_eq!(ncols, buf_x.len());
    assert_eq!(nrows, buf_fx.len());

    // Iterate over all columns
    jac.chunks_mut(nrows).enumerate().for_each(|(col, chunk)| {
        // Calculate x + h for column col
        buf_x.copy_from_slice(x);
        buf_x[col] = x[col] + h;

        // Evaluate function(x + h * delta)
        f(buf_fx, buf_x);

        chunk
            .iter_mut()
            .zip(buf_fx.iter())
            .for_each(|(output, input)| {
                *output = reciprocal * *input;
            });

        // Calculate x - h * delta_col
        buf_x.copy_from_slice(x);
        buf_x[col] = x[col] - h;

        // Evaluate function(x - h * delta)
        f(buf_fx, buf_x);

        chunk
            .iter_mut()
            .zip(buf_fx.iter())
            .for_each(|(output, input)| {
                *output -= reciprocal * *input;
            });
    })
}

#[cfg(test)]
mod tests {

    use super::*;
    use approx;

    #[test]
    fn test_jacobian_with_closure() {
        // Derive a 3x2 Jacobian
        let closure = |y: &mut [f64], x: &[f64]| {
            y[0] = x[0].powi(2);
            y[1] = x[1].powi(2);
            y[2] = x[0] * x[1]
        };

        let mut jacobian = vec![0.0; 6];

        // Calculate the Jacobian for the input x = [1.0, 2.0]

        // Serial version
        let x = [1.0, 2.0];

        let ncols = x.len();
        let nrows = jacobian.len() / ncols;

        let mut buf_x = vec![0.0; ncols];
        let mut buf_fx = vec![0.0; nrows];

        central_jacobian(
            jacobian.as_mut_slice(),
            x.as_slice(),
            std::f64::EPSILON.sqrt(),
            buf_x.as_mut_slice(),
            buf_fx.as_mut_slice(),
            closure,
        );
        approx::assert_abs_diff_eq!(jacobian[0], 2.0, epsilon = 1e-15);
        approx::assert_abs_diff_eq!(jacobian[1], 0.0, epsilon = 1e-15);
        approx::assert_abs_diff_eq!(jacobian[2], 2.0, epsilon = 1e-15);
        approx::assert_abs_diff_eq!(jacobian[3], 0.0, epsilon = 1e-15);
        approx::assert_abs_diff_eq!(jacobian[4], 4.0, epsilon = 1e-15);
        approx::assert_abs_diff_eq!(jacobian[5], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_jacobian_with_fn() {
        // Derive a 3x2 Jacobian
        fn function(y: &mut [f64], x: &[f64]) {
            y[0] = x[0].powi(2);
            y[1] = x[1].powi(2);
            y[2] = x[0] * x[1]
        }

        let mut jacobian = vec![0.0; 6];

        // Calculate the Jacobian for the input x = [1.0, 2.0]

        // Serial version
        let x = [1.0, 2.0];

        let ncols = x.len();
        let nrows = jacobian.len() / ncols;

        let mut buf_x = vec![0.0; ncols];
        let mut buf_fx = vec![0.0; nrows];

        central_jacobian(
            jacobian.as_mut_slice(),
            x.as_slice(),
            std::f64::EPSILON.sqrt(),
            buf_x.as_mut_slice(),
            buf_fx.as_mut_slice(),
            function,
        );
        approx::assert_abs_diff_eq!(jacobian[0], 2.0, epsilon = 1e-15);
        approx::assert_abs_diff_eq!(jacobian[1], 0.0, epsilon = 1e-15);
        approx::assert_abs_diff_eq!(jacobian[2], 2.0, epsilon = 1e-15);
        approx::assert_abs_diff_eq!(jacobian[3], 0.0, epsilon = 1e-15);
        approx::assert_abs_diff_eq!(jacobian[4], 4.0, epsilon = 1e-15);
        approx::assert_abs_diff_eq!(jacobian[5], 1.0, epsilon = 1e-15);
    }
}
