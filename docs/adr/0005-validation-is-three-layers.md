---
status: accepted
---

# Validation is three layers, each at the scale where it has power

Correctness is checked by three instruments, deliberately used at different
scales, because no single one has power everywhere.

**Design identities**, no simulation, any N. The π₂ matrix must satisfy
`Σ_l π_kl = n·π_k` for a fixed-size design, be symmetric, and carry π on its
diagonal. These are exact algebra, cost nothing, and catch mathematical errors
immediately.

**Cell-wise π̂₂ against the analytic π₂**, small N only. Comparing an N×N matrix
entry by entry needs `B ≫ N²` replicates to have any power, so this runs at
N=20–40 with very large B, where it is unforgiving.

**Sen–Yates–Grundy functional**, high N and n, **fixed-size designs only**. At
N=400 the π₂ matrix has 160k cells and cell-wise comparison is statistically
hopeless. Sen–Yates–Grundy collapses the whole matrix into one scalar variance
for a given `y`, which Monte Carlo estimates precisely with modest B. Checked
for a spanning set of `y` — proportional to π, orthogonal to π, random,
adversarial — alongside Horvitz–Thompson unbiasedness.

The fixed-size restriction is not a detail. Sen–Yates–Grundy is derived by using
`Σ_k S_k = n` as a constant, so it is simply invalid for a random-size design,
and it fails *silently*: for Poisson every off-diagonal `Δ_kl` is zero, so the
formula returns **0.00** where the true variance is 1622. A plausible number
that is completely wrong is the worst failure mode a test battery can have.

Each design therefore declares `fixed_size`, and `var_syg` raises on a
random-size design rather than returning zero. Two variance forms cover
everything: the double sum `ΣΣ Δ_kl y_k y_l / (π_k π_l)` is valid for **any**
design, and Poisson has the closed form `Σ (1 − π_k) y_k² / π_k`, which needs no
π₂ at all.

## Consequences

The estimator layer becomes public: `package_sampling.estimators` with
`ht_total`, `var_ht`, `var_syg`, `var_deville` and `hajek_total`. These are not
new machinery — they are four functions of R's `sampling` that were never
ported (`HTestimator`, `varHT`, `varest`, `Hajekestimator`), and `varHT` with
`method=2` *is* the Sen–Yates–Grundy form the battery needs. Making them public
closes a real gap in the port and lets the battery test designs through the
published interface rather than through private test helpers.

The high-N layer costs minutes, not seconds, even after ADR-0002. It runs behind
`@pytest.mark.slow` in CI; the default `pytest` keeps the fast layers.
