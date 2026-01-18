"""
WILDKATZE-I: Military Language Model for Psychological Operations

This package provides the core implementation of the WILDKATZE-I psyLM,
a 28B parameter transformer model specialized for psychological operations
and strategic communication research.

Copyright (c) 2026 olaflaitinen. All rights reserved.
Licensed under EUPL v1.2
"""

__version__ = "1.0.0"
__author__ = "Olaf Laitinen"
__email__ = "research@wildkatze.mil"

from wildkatze.model.config import WildkatzeConfig
from wildkatze.model.architecture import WildkatzeModel, WildkatzeForCausalLM

__all__ = [
    "WildkatzeConfig",
    "WildkatzeModel",
    "WildkatzeForCausalLM",
]
