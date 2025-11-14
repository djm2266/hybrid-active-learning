#!/usr/bin/env python3
"""Evaluation metrics"""
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from typing import List, Dict


def compute_metrics(y_true: List, y_pred: List) -> Dict:
    """Compute classification metrics"""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    }
