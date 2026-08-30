"""Parity with R's *sampling* 2.10, proven against a recorded corpus.

ADR-0001 makes R the specification; this is where that claim is checked.
Random designs are compared value-for-value by replaying R's own uniform
stream (ADR-0006), not merely in distribution.

Cases named ``ERR_*`` record inputs where **R raises**.  Some of those are
upstream defects our port deliberately survives — see ADR-0007.
"""

from __future__ import annotations

import numpy as np
import pytest

from package_sampling.sampling import (
    UPBrewer,
    UPMaxEntropy,
    UPPoisson,
    UPSystematic,
    up_max_entropy_pi2,
    up_systematic_pi2,
    up_tille_pi2,
    upme_pik_tilde_from_pik,
    upme_q_from_w,
)
from package_sampling.utils import inclusion_probabilities
from tests.golden_corpus import ReplayRng, load_cases

CASES = load_cases()

DESIGNS = {
    "UPbrewer": UPBrewer,
    "UPtille": None,  # bound below, avoids a circular import at module load
    "UPsystematic": UPSystematic,
    "UPpoisson": UPPoisson,
    "UPmaxentropy": UPMaxEntropy,
}


def _design_cases():
    return [n for n in CASES if n.split("_")[0] in DESIGNS and not n.startswith("ERR_")]


def test_corpus_is_present_and_covers_every_design():
    assert CASES, "golden corpus is empty"
    covered = {n.split("_")[0] for n in CASES}
    for fn in ("UPbrewer", "UPtille", "UPsystematic", "UPpoisson", "UPmaxentropy"):
        assert fn in covered, f"{fn} has no golden"


# --------------------------------------------------------------- replayed
@pytest.mark.parametrize("name", _design_cases())
def test_random_designs_replay_r_exactly(name):
    from package_sampling.sampling import UPTille

    classes = dict(DESIGNS, UPtille=UPTille)
    case = CASES[name]
    assert case.u is not None and case.out is not None

    rng = ReplayRng(case.u)
    got = classes[name.split("_")[0]](case.pik).draw(rng)

    np.testing.assert_array_equal(
        got, case.out.astype(np.int8), err_msg=f"{name} diverges from R"
    )
    # Drawing too few uniforms would otherwise pass silently; asking for too
    # many already raises inside ReplayRng. Together they pin the consumption
    # count, which is what makes replay meaningful.
    assert rng.exhausted, (
        f"{name}: consumed {rng._i} of {case.u.size} uniforms; the port and R "
        f"step through the random stream differently."
    )


# ----------------------------------------------------------- deterministic
@pytest.mark.parametrize(
    "prefix, fn",
    [
        ("UPtillepi2", up_tille_pi2),
        ("UPsystematicpi2", up_systematic_pi2),
        ("UPmaxentropypi2", up_max_entropy_pi2),
    ],
)
def test_pi2_matrices_match_r(prefix, fn):
    seen = 0
    for name, case in CASES.items():
        if not name.startswith(prefix + "_"):
            continue
        seen += 1
        np.testing.assert_allclose(
            fn(case.pik), case.matrix, rtol=1e-9, atol=1e-12, err_msg=name
        )
    assert seen, f"no goldens for {prefix}"


def test_pik_tilde_matches_r():
    seen = 0
    for name, case in CASES.items():
        if not name.startswith("UPMEpiktildefrompik_"):
            continue
        seen += 1
        np.testing.assert_allclose(
            upme_pik_tilde_from_pik(case.pik), case.out, rtol=1e-6, atol=1e-9
        )
    assert seen


@pytest.mark.parametrize("name", ["UPMEqfromw_w123_n2", "UPMEqfromw_w1234_n3"])
def test_q_from_w_matches_r(name):
    case = CASES[name]
    n = int(case.extra[0])

    np.testing.assert_allclose(
        upme_q_from_w(case.pik, n), case.matrix, rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize(
    "name", ["inclusionprobabilities_basic", "inclusionprobabilities_capped"]
)
def test_inclusion_probabilities_match_r(name):
    case = CASES[name]
    n = int(case.extra[0])

    np.testing.assert_allclose(
        inclusion_probabilities(case.pik, n), case.out, rtol=1e-12, atol=1e-14
    )


# ------------------------------------------------- inputs where R raises
def test_every_error_case_is_recorded_as_such():
    errs = {n: c.error for n, c in CASES.items() if n.startswith("ERR_")}
    assert errs, "no R-raises goldens"
    for name, msg in errs.items():
        assert msg, f"{name} is marked ERR_ but records no message"


@pytest.mark.parametrize(
    "name, behaviour",
    [
        # R crashes; we return a sample (ADR-0007)
        ("ERR_UPMEqfromw_n1", "ours works"),
        ("ERR_UPmaxentropy_subeps", "ours works"),
        ("ERR_UPmaxentropy_alldet", "ours works"),
        # R raises and so do we, for our own reasons
        ("ERR_UPtillepi2_na", "ours raises"),
        ("ERR_UPbrewer_alldet", "ours raises"),
    ],
)
def test_r_error_cases_behave_as_the_adrs_say(name, behaviour):
    from package_sampling.sampling import UPTille  # noqa: F401

    case = CASES[name]
    assert case.raises, f"{name} should record an R error"

    runners = {
        "ERR_UPMEqfromw_n1": lambda: upme_q_from_w(case.pik, 1),
        "ERR_UPmaxentropy_subeps": lambda: UPMaxEntropy(case.pik).draw(
            np.random.default_rng(0)
        ),
        "ERR_UPmaxentropy_alldet": lambda: UPMaxEntropy(case.pik).draw(
            np.random.default_rng(0)
        ),
        "ERR_UPtillepi2_na": lambda: up_tille_pi2(case.pik),
        "ERR_UPbrewer_alldet": lambda: UPBrewer(case.pik),
    }

    if behaviour == "ours works":
        assert runners[name]() is not None
    else:
        with pytest.raises(ValueError):
            runners[name]()
