"""Error-Gated PNN Maturation -- the novel continual-learning method."""
from .closure import ClosureDetector
from .consolidation import PNNConsolidation
from .importance import ImportanceEstimator
from .method import PNNMethod

__all__ = ["PNNMethod", "PNNConsolidation", "ClosureDetector", "ImportanceEstimator"]
