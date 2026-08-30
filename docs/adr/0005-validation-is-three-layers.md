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

**Sen–Yates–Grundy functional**, high N and n. At N=400 the π₂ matrix has 160k
cells and cell-wise comparison is statistically hopeless. Sen–Yates–Grundy
collapses the whole matrix into one scalar variance for a given `y`, which Monte
Carlo estimates precisely with modest B. Checked for a spanning set of `y` —
proportional to π, orthogonal to π, random, adversarial — alongside
Horvitz–Thompson unbiasedness.

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
