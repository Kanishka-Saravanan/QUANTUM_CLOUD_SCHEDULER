"""
Classical cloud scheduling algorithms.
"""

from .fifo import FIFOScheduler
from .round_robin import RoundRobinScheduler

__all__ = ["FIFOScheduler", "RoundRobinScheduler"]
