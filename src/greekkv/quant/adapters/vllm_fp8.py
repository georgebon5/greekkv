"""vLLM FP8 KV cache — **OPTIONAL**, skipped on free-tier hardware.

vLLM's FP8 KV cache is a serving-engine feature, not a pluggable transformers
``Cache`` subclass. Using it requires a vLLM engine on a GPU with FP8 support
(H100 / Ada), which the project does not have on free hardware.

This adapter exists for completeness of the ``scheme`` dispatch: when the
eval harness runs with ``scheme="vllm-fp8"`` we raise immediately so no
numbers end up in ``results/`` labelled as **measured**. The capacity model
(T28) is allowed to produce ``vllm-fp8`` rows but they are explicitly tagged
as **estimate**.
"""

from __future__ import annotations

from typing import Any

_HINT = (
    "'vllm-fp8' is OPTIONAL and requires a GPU with FP8 support + vLLM engine. "
    "On free-tier hardware this scheme is only meaningful in the T28 capacity "
    "model where its outputs are labelled 'estimate', not 'measured'."
)


def build_vllm_fp8(spec: Any) -> Any:
    raise NotImplementedError(_HINT)
