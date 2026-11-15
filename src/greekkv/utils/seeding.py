from __future__ import annotations

import os
import random


def seed_all(seed: int) -> None:
    """Seed every RNG we care about.

    Optional deps (numpy, torch, transformers) are seeded only if importable,
    so this function stays usable from the scaffold with no heavy deps.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    try:
        import transformers  # type: ignore[import-not-found]
    except ImportError:
        pass
    else:
        transformers.set_seed(seed)
