"""
Brain Tumor MRI Classification — modular training pipeline.

Public API:
    from src import Config, build_dataloaders, build_model, train_model, evaluate_model
"""
from .config import CFG, Config, ensure_dirs
from .custom_cnn import CustomBrainCNN
from .data import build_dataloaders, compute_class_weights
from .engine import train_model
from .evaluate import (evaluate_model, inference_time_benchmark, mcnemar_test,
                       plot_history, plot_misclassification_deepdive)
from .losses import FocalLoss, build_criterion
from .models import TemperatureScaling, build_model, ece_score
from .utils import describe_splits, save_run_config, set_seed

__all__ = [
    "CFG", "Config", "ensure_dirs",
    "build_dataloaders", "compute_class_weights",
    "build_model", "train_model",
    "evaluate_model", "plot_history",
    "set_seed", "save_run_config", "describe_splits",
    "CustomBrainCNN", "FocalLoss", "build_criterion",
    "TemperatureScaling", "ece_score",
    "inference_time_benchmark", "mcnemar_test", "plot_misclassification_deepdive",
]
