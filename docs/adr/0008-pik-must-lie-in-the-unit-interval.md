---
status: accepted
---

# pik must lie in the unit interval

Every design rejects a `pik` vector containing a value outside `[0, 1]`, and
the error names `inclusion_probabilities` as the repair. R's *sampling* does
not check, so this is a deliberate divergence under ADR-0001 — but unlike most
such calls it rests on a theorem rather than a preference.

`π_k ≡ P(k ∈ S)` is a probability, so `π_k ∈ [0, 1]` by definition. A vector
containing 1.143 is not an inclusion-probability vector; the object does not
exist.

Such vectors do arise legitimately. A πps design sets `π_k = n·x_k/Σx` from
size measures, and a large `x_k` pushes `π_k` past 1. The canonical remedy —
Tillé's own, and the one `inclusionprobabilities` implements in R and
`inclusion_probabilities` implements here — is the **take-all stratum**: cap
the offenders at 1, recompute the remainder over the reduced sample size, and
iterate to a fixed point. The repair already ships with this package, so
refusing the input costs the caller one function call.

## Why silence was not an option

Horvitz–Thompson unbiasedness requires `E[S_k] = π_k` for every unit. A design
given `π_k = 1.143` treats that unit as certain, so `E[S_k] = 1`, while the
estimator still divides by 1.143. The term is scaled by `1/1.143 = 0.875` and
the estimator is biased. Measured over 200,000 draws of `UPSystematic` with
`pik = [0.857, 1.0, 1.143]` and `y = [10, 50, 90]`:

```
requested pi : [0.857143  1.        1.142857]     E|S| requested = 3.0000
realised  pi : [0.858425  1.        1.      ]     E|S| realised  = 2.8584

true total Y      150.000
E[Y_HT]           138.765
bias               -11.235   (-7.49%)
```

A 7.5% bias, with no warning, in the single estimator the design exists to
keep unbiased.

## Consequences

The check lives in one place, `utils.check_pik`, rather than in five
constructors. `up_max_entropy` already validated the range while its four
siblings did not; routing all of them through the shared validator removes
that inconsistency as a side effect.

Two tests in `test_up_tille.py` reached the old `n == N` census branch only by
passing `[0.857, 1.0, 1.143]` — that is, by exploiting the missing check. They
were removed when the branch was deleted under ADR-0007, so nothing depends on
the old permissiveness.
