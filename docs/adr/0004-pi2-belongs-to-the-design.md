---
status: accepted
---

# π₂ belongs to the design

Every design exposes `.pi2()`. Previously the pairing between a design and its
joint inclusion probabilities existed only as a keyword argument at one call
site in the test suite, which meant nothing forced a design to be checked
against its own second-order structure.

`UPSystematic`, `UPTille` and `UPMaxEntropy` return the exact matrices ported
from R. `UPPoisson` gains one R never shipped: Poisson draws are independent, so
`π_kl = π_k π_l` off the diagonal — one line, exact, and it brings the design
into the cell-wise second-order test layer, where comparing π̂₂ against
`π_k π_l` empirically verifies the draws really are independent.
`UPBrewer.pi2()` raises `NotImplementedError`; Brewer's joint inclusion
probabilities have no known closed form, which is why R omits `UPbrewerpi2` too.

R's omission of `UPpoissonpi2` should be read as a judgement rather than a gap.
Poisson's π₂ is **degenerate**: `Δ = π₂ − ππ'` is identically zero off the
diagonal, so the N×N matrix holds nothing the N-vector π does not, and
Poisson's variance is a single sum needing no matrix at all. We ship it for a
uniform interface and for the independence test, not because it carries
information.

## Consequences

Identity expectations are per-design, not blanket assertions. The row-sum
identity `Σ_l π_kl = n·π_k` holds only for fixed-size designs, so not for
Poisson. The Yates–Grundy condition fails for systematic sampling by design —
that is why Sen–Yates–Grundy variance estimates can go negative under it, and a
test asserting it universally would be asserting something false.
