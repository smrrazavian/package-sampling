---
status: accepted
---

# Designs are built once and drawn many times

Each design becomes a class built from `pik` — `UPMaxEntropy(pik)` — that holds
the precomputed state and can be drawn from repeatedly via `.draw(rng)`. The
existing `up_*` functions remain as one-shot wrappers, so nothing published
breaks.

Almost all the cost of a draw is work that depends only on `pik` and was being
redone on every call. At N=2000, `up_max_entropy` spent 1512 ms in
`upme_pik_tilde_from_pik` and 436 ms in `upme_q_from_w` against **2.1 ms** in
the actual draw; `up_tille` spent 89% of each draw recomputing
`inclusion_probabilities`. A Monte-Carlo run of B=5000 went from ~2.7 hours to
~13 seconds. The statistical validation in ADR-0005 is not affordable at the
required scale without this.

## Considered options

Memoising the `pik → (π̃, q)` computation behind the existing functions was
rejected: NumPy arrays are unhashable, it introduces hidden global state with an
eviction policy to get right, and it leaves π₂ with nowhere to live. Letting
only the test suite reach past the interface to the `upme_*` helpers was
rejected because tests would then bypass the very interface they exist to test.

## Consequences

This reintroduces classes, which commit `9fcda79` (April 2025) deliberately
removed. That removal was correct and is not being reversed in spirit: the
deleted `SamplingMethod` was an abstract base with one method and **no state** —
nothing a function could not do. These classes exist precisely to hold state
(π̃, q, the live mask, n) across draws. No ABC, no inheritance, one concrete
class per design.

The shape follows `scipy.stats` frozen distributions — `norm(mu, sigma).rvs()` —
which is the pattern a Python user of a sampling library already expects.
