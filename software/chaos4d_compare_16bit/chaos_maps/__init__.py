# Maintainer: zhaowenbo
"""Chaotic map implementations."""

from chaos_maps.hccdm_4d import HCCDM4DMap
from chaos_maps.idhnn_4d import IDHNN4DMap
from chaos_maps.literature_high_dim import (
    CyclicSymmetricLogisticMap,
    SineMapLatticeMemristiveMap,
    YaoComplexPascalMap,
    ZangLinearModularMap,
)
from chaos_maps.mhm_4d import MHM4DMap
from chaos_maps.mprc_4d import MPRC4DMap
from chaos_maps.mslc_4d import MSLC4DMap
from chaos_maps.ndcs_4d import NDCS4DMap
from chaos_maps.proposed_base import Proposed4DBaseMap, ProposedNDBaseMap
from chaos_maps.proposed_enhanced import Proposed4DEnhancedMap, ProposedNDEnhancedMap
from chaos_maps.uruk_4d_dcm import URUK4DDCMMap

__all__ = [
    "Proposed4DBaseMap",
    "Proposed4DEnhancedMap",
    "ProposedNDBaseMap",
    "ProposedNDEnhancedMap",
    "NDCS4DMap",
    "URUK4DDCMMap",
    "MHM4DMap",
    "IDHNN4DMap",
    "MPRC4DMap",
    "MSLC4DMap",
    "HCCDM4DMap",
    "YaoComplexPascalMap",
    "SineMapLatticeMemristiveMap",
    "CyclicSymmetricLogisticMap",
    "ZangLinearModularMap",
]
