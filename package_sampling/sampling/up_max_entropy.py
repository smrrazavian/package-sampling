"""
Maximum-Entropy sampling design  (R `UPmaxentropy` → Python).

The design draws a *fixed-size* sample that

  • matches the required first-order inclusion probabilities ``pik``;
  • maximises Shannon entropy subject to that constraint.

Internally it uses the sequential Max-Entropy machinery

        π̃  →  w  →  q  →  s₂

where
    wᵢ   = π̃ᵢ / (1 − π̃ᵢ) ,
    q    = upme_q_from_w(w, n) and
    s₂   = upme_s_from_q(q)

(see Tillé & Matei, 2016, Chap. 11).

Deterministic units
-------------------
• πᵢ ≥ 1 − *eps* → always selected
• πᵢ ≤ *eps*   → never selected

Only the “live’’ units (eps < πᵢ < 1−eps) drive the Max-Entropy loop.

"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Union

from package_sampling.utils import as_int
from .upme_pik_tilde_from_pik import upme_pik_tilde_from_pik
from .upme_q_from_w import upme_q_from_w
from .upme_s_from_q import upme_s_from_q


def up_max_entropy(
    pik: Union[List[float], NDArray[np.floating]],
    eps: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int8]:
    """
    Maximum-Entropy unequal-probability sampling without replacement.

    Parameters
    ----------
    pik : 1-D array-like of float
        Target inclusion probabilities (0 ≤ πᵢ ≤ 1).
    eps : float, default 1e-6
        Tolerance that separates deterministic 0/1 units from “live’’ ones.
    rng : numpy.random.Generator | None
        Random source.  A fresh ``default_rng()`` is created if *None*.

    Returns
    -------
    sel : ndarray[int8]
        0/1 selection mask with ``dtype == np.int8`` and
        ``sel.sum() == round(sum(pik))``.
    """
    # ------------- input coercion & sanity checks -------------------------
    if not isinstance(pik, np.ndarray):
        pik = np.asarray(pik, dtype=float)
    if pik.ndim != 1:
        raise ValueError("pik must be a 1-D vector")
    if np.any((pik < 0) | (pik > 1) | np.isnan(pik)):
        raise ValueError("Inclusion probabilities must be between 0 and 1.")
    if pik.size == 0:
        return np.zeros(0, dtype=np.int8)

    rng = rng or np.random.default_rng()
    n_tot = as_int(round(pik.sum()))

    # ------------- deterministic selections ------------------------------
    sel = np.zeros_like(pik, dtype=np.int8)
    sel[pik >= 1 - eps] = 1
    if n_tot == 0:
        return sel
    if n_tot == sel.sum():
        return sel

    # ---------------------------------------------------------------------
    # CASE 1 : n_tot == 1  → simple multinomial draw
    # ---------------------------------------------------------------------
    if n_tot == 1:
        probs = pik / pik.sum()
        idx = rng.choice(pik.size, p=probs)
        sel[idx] = 1
        return sel

    # ---------------------------------------------------------------------
    # CASE 2 : n_tot ≥ 2  → sequential Max-Entropy algorithm
    # ---------------------------------------------------------------------
    live_mask = (pik > eps) & (pik < 1 - eps)
    pik_live = pik[live_mask]
    n_live = n_tot - sel.sum()

    if n_live == 0:
        return sel

    # ------ fixed-point to obtain π̃ -------------------------------------
    pik_tilde = upme_pik_tilde_from_pik(pik_live, eps=eps)

    # ------ build q-matrix, draw sample ----------------------------------
    with np.errstate(divide="ignore"):
        w = pik_tilde / (1.0 - pik_tilde)
    q = upme_q_from_w(w, n_live)
    sel_live = upme_s_from_q(q)

    sel[live_mask] = sel_live.astype(np.int8)
    return sel
