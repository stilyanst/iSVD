"""Incremental truncated SVD for streaming matrices.

Companion code of the paper "Incremental SVD for Large-Scale Dynamic
Matrices" (arXiv:2605.24514).
"""
from .core import IncrementalSVD
from .policies import bound, periodic

__version__ = "1.0.0"

__all__ = ["IncrementalSVD", "periodic", "bound", "__version__"]
