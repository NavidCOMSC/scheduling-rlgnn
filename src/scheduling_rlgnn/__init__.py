"""
Scheduling RLGNN: GNN architecture models with PyG
and deep reinforcement learning.
"""

__version__ = "0.1.0"
__author__ = "Navid Rahimi"
__email__ = "amir.navid.rahimi@googlemail.com"

from . import envwrapper
from . import models

__all__ = [
    "envwrapper",
    "models",
    # "JobShopEnvironmentWrapper",
]
