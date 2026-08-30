"""
Sampling algorithms for unequal probability designs (UP* family).

This module exposes various probability-based sampling methods including:
- Systematic sampling
- Tillé’s method
- Max Entropy sampling
- Brewer’s method
- Pivotal methods and intermediate transformations (e.g., q-from-w,
  pik-tilde-from-pik)

Each algorithm returns a binary vector indicating selected units.
"""

from .up_brewer import UPBrewer, up_brewer  # noqa: F401
from .up_max_entropy import UPMaxEntropy, up_max_entropy  # noqa: F401
from .up_max_entropy_pi2 import up_max_entropy_pi2  # noqa: F401
from .up_poisson import UPPoisson, up_poisson  # noqa: F401
from .up_systematic import UPSystematic, up_systematic  # noqa: F401
from .up_systematic_pi2 import up_systematic_pi2  # noqa: F401
from .up_tille import UPTille, up_tille  # noqa: F401
from .up_tille_pi2 import up_tille_pi2  # noqa: F401
from .upme_pik2_from_pik_w import upme_pik2_from_pik_w  # noqa: F401
from .upme_pik_from_q import upme_pik_from_q  # noqa: F401
from .upme_pik_tilde_from_pik import upme_pik_tilde_from_pik  # noqa: F401
from .upme_q_from_w import upme_q_from_w  # noqa: F401
from .upme_s_from_q import upme_s_from_q  # noqa: F401

__all__ = [
    "up_brewer",
    "UPBrewer",
    "up_max_entropy",
    "UPMaxEntropy",
    "up_max_entropy_pi2",
    "up_systematic",
    "UPSystematic",
    "up_systematic_pi2",
    "up_tille",
    "UPTille",
    "up_tille_pi2",
    "up_poisson",
    "UPPoisson",
    "upme_pik_from_q",
    "upme_pik_tilde_from_pik",
    "upme_pik2_from_pik_w",
    "upme_q_from_w",
    "upme_s_from_q",
]

__version__ = "1.0.0"
