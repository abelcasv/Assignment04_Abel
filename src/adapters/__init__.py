"""
src.adapters — Non-invasive execution adapters for Assignment 1 and Assignment 3 baselines.
"""

from src.adapters.a1_adapter import A1SequentialAdapter
from src.adapters.a3_adapter import A3OpenMPAdapter

__all__ = ["A1SequentialAdapter", "A3OpenMPAdapter"]
