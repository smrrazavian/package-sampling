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
into the second-order test layer. `UPBrewer.pi2()` raises `NotImplementedError`;
Brewer's joint inclusion probabilities have no known closed form, which is why R
omits `UPbrewerpi2` too.

## Consequences

Identity expectations are per-design, not blanket assertions. The row-sum
identity `Σ_l π_kl = n·π_k` holds only for fixed-size designs, so not for
Poisson. The Yates–Grundy condition fails for systematic sampling by design —
that is why Sen–Yates–Grundy variance estimates can go negative under it, and a
test asserting it universally would be asserting something false.
