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

A third crash has a different cause. `UPmaxentropy` dispatches on `n` through
`if(n>=2) ... if(n==0) ... if(n==1)`, and a non-integer `n` matches none of
them, so the result variable is never assigned:

```
UPmaxentropy(c(1e-7, 5e-7, 2e-7))
  ->  Warning: the argument is not integer
  ->  Error: object 's' not found
```

Every unit there is below `eps`, so none is live and the sample is empty — a
well-defined design, and we return `[0, 0, 0]` for it. Note that our own
`UPMaxEntropy` derived `n` from the whole `pik` vector rather than the live
subset, which made it reject this input once the tolerance below was
tightened; `n` is now `n_certain + as_int(sum(pik_live))`, as this ADR
specifies.

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

### Where the boundary sits, and why

The tolerance is `τ = sqrt(machine epsilon) = 1.490116e-08`, applied relatively
except when the rounded target is zero, where it is absolute. That is exactly
what R's `.as_int` gets from `all.equal`, whose default tolerance is
`sqrt(.Machine$double.eps)` and which switches to an absolute comparison for a
near-zero target. Matching it keeps the accept/reject boundary identical to the
package we port, so a golden corpus agrees with R on which inputs are valid at
all.

It is also far above anything floating-point error can produce. NumPy sums
pairwise, bounded by `(log2 N + 1)·u·Σ|x|` with `u = 2⁻⁵³`, and the measured
relative error in `sum(pik)` stays near `2e-16` from N = 10 to N = 10⁶. At
Σπ = 200:

| tolerance | accepts ± | vs. measured error |
| --- | --- | --- |
| measured float error, N = 10⁶ | 4.6e-13 | 1× |
| R, and now us — relative `sqrt(eps)` | 2.98e-06 | ~6×10³ |
| our previous `np.isclose(atol=1e-6)` | 2.00e-03 | ~4×10⁶ |

The previous value was not chosen so much as inherited: `np.isclose` carries a
default `rtol=1e-5`, so the window widened with the magnitude of the sum and
`200.0015` passed — a value R rejects. ADR-0007 was therefore enforced at small
`n` and effectively unenforced at large `n`.

One structural consequence. `n == 0` requires `Σπ_live ≤ τ`, while a live unit
has `π > eps`, so the branch is reachable only when a caller sets `eps < τ`. At
the default `eps = 1e-6 > τ` it cannot be reached. Under the old tolerance it
was reachable through a razor-thin window that existed only because
`np.isclose`'s `atol` and `eps` were both `1e-6` — two unrelated constants
colliding, not a designed property.

## Missing values are rejected everywhere

`UPtillepi2` warns and then dies on an unhelpful internal condition:

```
UPtillepi2(c(0.2, NA, 0.5, 0.3))
  ->  Warning: there are missing values in the pik vector
  ->  Error: missing value where TRUE/FALSE needed
```

Our `up_tille_pi2` had no check at all. It gains one, raising `ValueError`
like its siblings, which matches R's intent — R clearly means to reject the
input, it just does so by tripping over an internal condition.

The *check* is now shared: every design routes through `utils.check_pik`,
which also enforces the `[0, 1]` range (ADR-0008). The *wording* is not.
`check_pik` takes the message per design, so five spellings remain:

```
up_brewer       "Missing values detected in the pik vector."
up_tille        "`pik` contains NaN values."
up_tille_pi2    "`pik` contains NaN values."
up_systematic   "There are missing values in the pik vector"
up_poisson      "There are missing values in the `pik` vector."
up_max_entropy  "Inclusion probabilities must be between 0 and 1."
```

They were kept because published tests match on them. That is a weak reason
and the last one is actively misleading — a NaN gets told it is out of range.
Unifying the wording is a breaking change for anyone matching on the text, so
it belongs in the 2.0.0 release rather than here.

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
