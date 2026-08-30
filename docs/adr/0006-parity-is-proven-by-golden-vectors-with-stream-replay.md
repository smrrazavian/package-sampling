---
status: accepted
---

# Parity is proven by golden vectors with stream replay

ADR-0001 makes R the specification, but nothing in this repository checked that
claim — every test was self-referential, asserting statistical properties rather
than agreement with R. Parity is now proven by a corpus of goldens generated
from R's **sampling** 2.10 in a container and checked in as data.

Agreement is exact, including for the random designs. Each Python design already
consumes uniform random numbers one-for-one with its R counterpart, in the same
order — Brewer draws `n`, Tillé draws `N−n`, systematic draws 1, max-entropy
draws at most `N` — so replaying R's recorded `runif` values through `.draw()`
compares a Python sample to an R sample value-for-value.

That exactness matters because Python carries guard branches R has no equivalent
of: the `eps` floor on Brewer's denominator, the `total == 0` uniform fallbacks
in Brewer and Tillé, and Tillé's normalisation of a cumulative vector R leaves
unnormalised. These fire only on near-degenerate `pik`, which random testing
essentially never reaches and distributional tests cannot see at all. The corpus
includes inputs chosen to trigger them.

## Consequences

A golden can record that **R raises**. R crashes on inputs this package handles
(see ADR-0007), so the corpus needs three categories, not two: R's value, R's
error, and the deliberate divergences.

Generating goldens needs R; checking them does not. Generation runs in a
container as a separate workflow, so the ordinary test run has no R dependency.
Only pure-R sources are needed — `MASS` and `lpSolve` are required by `varest`,
`gencalib` and `UPminimalsupport`, none of which the corpus covers — so the
package need not be installed at all, only sourced.
