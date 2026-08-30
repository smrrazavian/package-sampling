---
status: accepted
---

# Design classes keep the UP prefix

The design classes are named `UPBrewer`, `UPTille`, `UPSystematic`,
`UPMaxEntropy`, `UPPoisson` — not `Brewer`, `Tille`, `Systematic`. `UP` stands
for unequal probability, and it marks a family rather than decorating a name.

This looks redundant today, when unequal-probability designs are all there is.
It stops looking redundant as soon as the equal-probability family lands:
R ships `srswor`, `srswr` and `srswor1`, which take `(n, N)` rather than a `pik`
vector, and `Systematic` alone becomes ambiguous the moment equal-probability
systematic sampling exists. R's UP family also has eight designs still unported
— `UPmidzuno`, `UPminimalsupport`, `UPmultinomial`, `UPopips`, `UPpivotal`,
`UPrandompivotal`, `UPrandomsystematic`, `UPsampford` — so the namespace is
heading for roughly sixteen designs across two families.

The second reason is the users: many of them read R's documentation. Someone who
knows `UPtille` finds `UPTille` by eye.

## Considered options

Plain names (`MaxEntropy`, `Poisson`) read best at a call site and were the
first recommendation, until the two-family future made the prefix load-bearing.
Family subpackages (`up.Tille`, `srs.WOR`) are more idiomatic Python but break
the R user's search. Exact R spelling (`UPtille`) maximises recognisability at
the cost of every linter flagging every class name.
