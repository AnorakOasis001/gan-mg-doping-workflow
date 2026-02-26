# Numerical stability in canonical thermodynamics

When computing canonical thermodynamics, we evaluate the partition function:

\[
Z = \sum_i \exp\left(-\frac{\Delta E_i}{k_B T}\right)
\]

A direct implementation can be numerically unstable:

- **Overflow** can occur if exponent arguments are very large positive values.
- **Underflow** can occur if exponent arguments are very large negative values.

To avoid this, the code evaluates `log(Z)` using the log-sum-exp identity.
Let:

\[
x_i = -\frac{\Delta E_i}{k_B T}, \quad m = \max_i x_i
\]

Then:

\[
\log Z = m + \log\left(\sum_i \exp(x_i - m)\right)
\]

This is algebraically equivalent to the original expression, but keeps the exponential arguments close to zero, which improves floating-point robustness.

The thermodynamic free energy contribution is then computed from `log(Z)` as:

\[
F = -k_B T \log Z
\]

This approach preserves existing CLI/API outputs while improving numerical reliability across wider energy and temperature ranges.
