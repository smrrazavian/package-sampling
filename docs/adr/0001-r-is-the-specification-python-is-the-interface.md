---
status: accepted
---

# R is the specification, Python is the interface

This package ports R's **sampling** 2.10, whose authors are survey-sampling
mathematicians. We treat the numbers R produces as the specification: for the
same input, this package returns what R returns. What we may change freely is
everything a caller touches but no result depends on — calling convention,
return dtype, error type and message, and any capability Python can offer that R
cannot. Where R is genuinely wrong we fix it, deliberately and in writing (see
ADR-0007); we do not "improve" its mathematics on our own judgement.

## Consequences

Two axes, two authorities. A proposal that changes a returned value needs
evidence that R is wrong. A proposal that changes only how the function is
called needs only that it makes the package better to use from Python.

The `rng` parameter is the model case: R has no such argument anywhere in the
UP family — every function draws from R's single global stream. Adding `rng` is
pure interface, changes no distribution, and is the kind of divergence this ADR
licenses without argument.
