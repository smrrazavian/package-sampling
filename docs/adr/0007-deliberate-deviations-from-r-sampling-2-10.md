---
status: accepted
---

# Deliberate deviations from R sampling 2.10

ADR-0001 makes R the specification. These are the places we knowingly return
something else. Each was verified by running R's `sampling` 2.10 sources, not by
reading them. Nothing here should be "fixed" back toward R without reopening
this ADR.

## R crashes; we do not

`UPMEqfromw(w, n)` is broken for `n = 1`. Its inner loop is
`for(z in 2:min(N-i, n))`, and with `n = 1` that evaluates `2:1` → `c(2, 1)`, so
`z = 2` indexes a one-column matrix:

```
UPMEqfromw(c(1,2,3), n=1)   ->  Error: subscript out of bounds
UPMEqfromw(c(1,2,3), n=2)   ->  fine
```

This is the classic R `:` pitfall, not a mathematical choice. It propagates:
`UPmaxentropy(c(1e-9, 0.4, 0.6, 1.0))` and `UPmaxentropy(c(0, 1, 1, 0))` both
crash the same way. Our `upme_q_from_w` short-circuits `n == 1` to the closed
form and `up_max_entropy` routes a total of one to a multinomial draw, so both
inputs return a sample. We keep our behaviour.

## R leaks π into the sample indicator; we return a mask

`UPbrewer`, `UPtille` and `UPsystematic` build their result as `s = pik` and
then overwrite only the live positions, so a deterministic unit keeps its
original floating-point π in what is supposed to be a 0/1 membership vector:

```
UPbrewer(c(1e-9, 0.4, 0.6, 1-1e-9))
  ->  1e-09  0  1  0.999999999      (typeof: double)
```

`UPpoisson` in the same package returns a clean mask, so R is not consistent
with itself here. We return `int8` zeros and ones from every design. This also
settles `up_poisson`, which previously returned `int32` while its four siblings
returned `int8`.

## Non-integer Σπ over live units is an error, not a rounded guess

R's `.as_int` warns and returns the value **unrounded**:

```
.as_int(2.02)  ->  2.02   (is.integer: FALSE, with a warning)
```

`UPtille` and `UPtillepi2` then use the raw `sum(pikb)` as a loop bound, so
`1:(N-n)` with `N-n = 0.2` yields `1` — one iteration — and
`UPtillepi2(c(0.9, 0.9))` returns `diag(0.9, 0.9)` with a zero off-diagonal.
Our `as_int` rounds, which sent the same input down a census shortcut and
produced an all-ones matrix.

Both answers are artifacts. A fixed-size design over two units cannot have
sample size 1.8, and R's result comes from the behaviour of its range operator
rather than from the design. So we reject the input: rounding stays for sums
within tolerance of an integer, because `Σπ = 24.999999999999996` should
certainly give 25, and a genuinely non-integer sum now raises `ValueError`
instead of warning. This is what `as_int` was already detecting; the warning
simply becomes an error.

The all-ones matrix in `up_tille_pi2` becomes unreachable rather than fixed.

## Missing values are rejected everywhere

`UPtillepi2` warns and then dies on an unhelpful internal condition:

```
UPtillepi2(c(0.2, NA, 0.5, 0.3))
  ->  Warning: there are missing values in the pik vector
  ->  Error: missing value where TRUE/FALSE needed
```

Our `up_tille_pi2` had no check at all. It gains one, raising the same
`ValueError` as its siblings — matching R's intent, and matching the rest of
this package, with one message instead of the three spellings we had.

## Where we already match, and keep matching

R's designs disagree with each other on all-deterministic input, and we
reproduce that disagreement exactly, because it follows from each design's own
structure:

| input `c(0, 1, 1, 0)` | R | here |
| --- | --- | --- |
| Brewer | error | `ValueError` |
| Tillé | error | `ValueError` |
| systematic | `0 1 1 0` | `0 1 1 0` |
| Poisson | `0 1 1 0` | `0 1 1 0` |

`UPbrewer` and `UPtille` stop because every unit falls outside `(eps, 1-eps)`;
`UPsystematic` has no such check and does not need one. This is not drift to be
tidied away.
