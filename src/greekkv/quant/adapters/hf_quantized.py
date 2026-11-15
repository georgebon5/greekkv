"""Adapter for ``transformers.cache_utils.QuantizedCache`` (quanto / HQQ).

``QuantizedCache`` supports two backends: ``"quanto"`` and ``"HQQ"``. Each
requires an extra pip package (``optimum-quanto`` or ``hqq``) which we do
**not** pull into greekkv's runtime deps — installing them is per-experiment.

Design note vs T11 (KiviCache)
------------------------------
HF's ``QuantizedCache`` applies **per-channel** quantization to **both** keys
and values with a single ``q_group_size``. This is **not** exactly the KIVI
paper, which uses per-channel K + **per-token** V. Our :class:`KiviCache` in
T11 implements the paper; this adapter stays close to what HF ships so that
``scheme="quanto"`` and ``scheme="int4-kivi"`` are two distinct, comparable
rows in the eval harness.
"""

from __future__ import annotations

from typing import Any

_HINT_QUANTO = (
    "The 'quanto' cache backend requires `optimum-quanto`. Install with: pip install optimum-quanto"
)
_HINT_HQQ = "The 'HQQ' cache backend requires the `hqq` package. Install with: pip install hqq"


def build_hf_quantized(backend: str, model_config: Any, spec: Any) -> Any:
    """Return a configured ``QuantizedCache`` for the given backend.

    Raises ``ModuleNotFoundError`` with a pip hint if the optional backend
    package is not installed.
    """
    from transformers.cache_utils import QuantizedCache  # noqa: PLC0415

    # Normalize scheme string to QuantizedCache's backend name.
    if backend == "quanto":
        try:
            import optimum.quanto  # noqa: F401, PLC0415
        except ImportError as e:
            raise ModuleNotFoundError(_HINT_QUANTO) from e
        backend_name = "quanto"
    elif backend == "hqq":
        try:
            import hqq  # noqa: F401, PLC0415
        except ImportError as e:
            raise ModuleNotFoundError(_HINT_HQQ) from e
        backend_name = "HQQ"
    else:
        raise ValueError(f"unknown HF QuantizedCache backend: {backend!r}")

    # HF uses `None` to signal "no residual window"; our spec uses 0.
    residual = spec.residual if spec.residual > 0 else None

    return QuantizedCache(
        backend=backend_name,
        config=model_config,
        nbits=spec.bits,
        axis_key=0,
        axis_value=0,
        q_group_size=spec.group_size,
        residual_length=residual,
    )
