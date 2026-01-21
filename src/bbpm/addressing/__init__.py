"""BBPM addressing modules."""

from .bbpm_addressing import BBPMAddressing
from .block_selector import MASK64, mix64, select_block
from .prp_feistel import (
    derive_round_keys_vectorized,
    feistel_prp_vectorized,
    prp_offsets,
)

__all__ = [
    "BBPMAddressing",
    "select_block",
    "mix64",
    "MASK64",
    "prp_offsets",
    "feistel_prp_vectorized",
    "derive_round_keys_vectorized",
]
