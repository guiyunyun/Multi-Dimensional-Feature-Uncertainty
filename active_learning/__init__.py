"""
Active Learning Module for DINOv3
"""

from .feature_extractor import MultiLayerDINOv3
from .memory_bank import MemoryBank
from .uncertainty import UncertaintyEstimator
from .cascading_selector import CascadingSelector, Priority
from .cascading_selector_v1 import CascadingSelectorV1
from .cascading_selector_v2 import CascadingSelectorV2
from .classifier import SimpleClassifier, MLPClassifier
from .dataset import (
    ActiveLearningDataset,
    load_imagenet100,
    create_initial_labeled_pool
)
from .active_learner import ActiveLearner
from .prediction_uncertainty import PredictionUncertaintyEstimator, HybridUncertaintyFusion

__all__ = [
    'MultiLayerDINOv3',
    'MemoryBank',
    'UncertaintyEstimator',
    'CascadingSelector',
    'CascadingSelectorV1',
    'CascadingSelectorV2',
    'Priority',
    'SimpleClassifier',
    'MLPClassifier',
    'ActiveLearningDataset',
    'load_imagenet100',
    'create_initial_labeled_pool',
    'ActiveLearner',
    'PredictionUncertaintyEstimator',
    'HybridUncertaintyFusion',
]
