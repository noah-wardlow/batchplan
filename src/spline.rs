//! Uniform cubic B-splines, the shape of every planned path. Span `k` of a spline blends control
//! points `k..k + 4`; a path's first three and last three control points are equal, so it starts
//! and ends at rest. `basis` in kernels.wgsl mirrors `basis`.

/// Weights of the four control points of a span for the position at `u` in [0, 1].
#[inline]
pub(crate) fn basis(u: f32) -> [f32; 4] {
    let (u2, u3, v) = (u * u, u * u * u, 1.0 - u);
    [v * v * v / 6.0, (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0, (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0, u3 / 6.0]
}

/// Weights for the first derivative with respect to `u`.
#[inline]
pub(crate) fn basis_d1(u: f32) -> [f32; 4] {
    let v = 1.0 - u;
    [-0.5 * v * v, 0.5 * (3.0 * u * u - 4.0 * u), 0.5 * (-3.0 * u * u + 2.0 * u + 1.0), 0.5 * u * u]
}

/// Weights for the second derivative with respect to `u`.
#[inline]
pub(crate) fn basis_d2(u: f32) -> [f32; 4] {
    [1.0 - u, 3.0 * u - 2.0, 1.0 - 3.0 * u, u]
}

/// Weights for the third derivative, constant along a span.
pub(crate) const BASIS_D3: [f32; 4] = [-1.0, 3.0, -3.0, 1.0];

/// Where along span `s` (of `per_span`) collision is sampled during optimization.
#[inline]
pub(crate) fn sample_u(s: usize, per_span: usize) -> f32 {
    (s as f32 + 0.5) / per_span as f32
}

/// `Σ weights[i] * cp[span + i]` into `out`, for control points `cp` (`[points, dof]`).
#[inline]
pub(crate) fn blend(cp: &[f32], dof: usize, span: usize, weights: [f32; 4], out: &mut [f32]) {
    for (j, o) in out.iter_mut().enumerate().take(dof) {
        *o = (0..4).map(|i| weights[i] * cp[(span + i) * dof + j]).sum();
    }
}

/// The spline at `per_span` evenly spaced points of every span plus its exact end, written into
/// `out`, which holds `[(points - 3) * per_span + 1, dof]`.
pub(crate) fn dense(cp: &[f32], dof: usize, per_span: usize, out: &mut [f32]) {
    let spans = cp.len() / dof - 3;
    for span in 0..spans {
        for s in 0..per_span {
            let row = (span * per_span + s) * dof;
            blend(cp, dof, span, basis(s as f32 / per_span as f32), &mut out[row..row + dof]);
        }
    }
    out[spans * per_span * dof..].copy_from_slice(&cp[cp.len() - dof..]);
}

/// The point a fraction `phase` in [0, 1] of the way along the spline.
pub(crate) fn at_phase(cp: &[f32], dof: usize, phase: f32, out: &mut [f32]) {
    let spans = cp.len() / dof - 3;
    let x = phase.clamp(0.0, 1.0) * spans as f32;
    let span = (x as usize).min(spans - 1);
    blend(cp, dof, span, basis(x - span as f32), out);
}
