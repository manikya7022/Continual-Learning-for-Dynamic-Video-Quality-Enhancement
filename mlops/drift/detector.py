"""
Data Drift Detection for Model Monitoring.
Detects when input distribution differs from training data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from dataclasses import dataclass


@dataclass
class DriftResult:
    """Result of drift detection."""
    is_drift: bool
    score: float
    threshold: float
    method: str
    details: Dict = None


class DriftDetector:
    """
    Detect data drift using statistical tests.
    
    Methods:
        - MMD: Maximum Mean Discrepancy (for high-dim data)
        - KS: Kolmogorov-Smirnov test (per-feature)
        - PSI: Population Stability Index
    """
    
    def __init__(
        self,
        method: str = "mmd",
        threshold: float = 0.05,
        window_size: int = 1000,
    ):
        self.method = method
        self.threshold = threshold
        self.window_size = window_size
        
        self.reference_data: Optional[np.ndarray] = None
        self.current_window: List[np.ndarray] = []
    
    def set_reference(self, data: np.ndarray) -> None:
        """Set reference distribution from training data."""
        self.reference_data = data
    
    def update(self, sample: np.ndarray) -> Optional[DriftResult]:
        """
        Add new sample and check for drift.
        Returns DriftResult when window is full.
        """
        self.current_window.append(sample)
        
        if len(self.current_window) >= self.window_size:
            current_data = np.array(self.current_window)
            result = self.detect(current_data)
            self.current_window = []
            return result
        
        return None
    
    def detect(self, current_data: np.ndarray) -> DriftResult:
        """Detect drift between reference and current data."""
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        if self.method == "mmd":
            return self._mmd_test(current_data)
        elif self.method == "ks":
            return self._ks_test(current_data)
        elif self.method == "psi":
            return self._psi_test(current_data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _mmd_test(self, current: np.ndarray) -> DriftResult:
        """Maximum Mean Discrepancy test."""
        ref = self.reference_data
        
        # Flatten if needed
        if ref.ndim > 2:
            ref = ref.reshape(ref.shape[0], -1)
            current = current.reshape(current.shape[0], -1)
        
        # Subsample for efficiency
        n = min(500, len(ref), len(current))
        ref_sample = ref[np.random.choice(len(ref), n, replace=False)]
        cur_sample = current[np.random.choice(len(current), n, replace=False)]
        
        # Compute MMD with RBF kernel
        gamma = 1.0 / ref_sample.shape[1]
        
        def rbf_kernel(X, Y):
            XX = np.sum(X**2, axis=1, keepdims=True)
            YY = np.sum(Y**2, axis=1, keepdims=True)
            distances = XX + YY.T - 2 * X @ Y.T
            return np.exp(-gamma * distances)
        
        K_xx = rbf_kernel(ref_sample, ref_sample)
        K_yy = rbf_kernel(cur_sample, cur_sample)
        K_xy = rbf_kernel(ref_sample, cur_sample)
        
        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        
        return DriftResult(
            is_drift=mmd > self.threshold,
            score=mmd,
            threshold=self.threshold,
            method="mmd",
        )
    
    def _ks_test(self, current: np.ndarray) -> DriftResult:
        """Kolmogorov-Smirnov test per feature."""
        ref = self.reference_data.reshape(len(self.reference_data), -1)
        cur = current.reshape(len(current), -1)
        
        p_values = []
        for i in range(ref.shape[1]):
            _, p = stats.ks_2samp(ref[:, i], cur[:, i])
            p_values.append(p)
        
        # Use Bonferroni correction
        min_p = min(p_values) * len(p_values)
        
        return DriftResult(
            is_drift=min_p < self.threshold,
            score=min_p,
            threshold=self.threshold,
            method="ks",
            details={'p_values': p_values},
        )
    
    def _psi_test(self, current: np.ndarray) -> DriftResult:
        """Population Stability Index."""
        ref = self.reference_data.flatten()
        cur = current.flatten()
        
        # Create bins from reference
        bins = np.percentile(ref, np.arange(0, 101, 10))
        bins = np.unique(bins)
        
        # Compute proportions
        ref_counts, _ = np.histogram(ref, bins=bins)
        cur_counts, _ = np.histogram(cur, bins=bins)
        
        ref_props = ref_counts / len(ref) + 1e-10
        cur_props = cur_counts / len(cur) + 1e-10
        
        # PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        # Rule of thumb: PSI > 0.2 indicates significant drift
        psi_threshold = 0.2
        
        return DriftResult(
            is_drift=psi > psi_threshold,
            score=psi,
            threshold=psi_threshold,
            method="psi",
        )


class ModelDriftMonitor:
    """
    Monitor model performance drift.
    Triggers retraining when quality degrades.
    """
    
    def __init__(
        self,
        metric_threshold: float = 0.1,
        window_size: int = 100,
    ):
        self.metric_threshold = metric_threshold
        self.window_size = window_size
        
        self.baseline_metric: Optional[float] = None
        self.metric_history: List[float] = []
    
    def set_baseline(self, metric: float) -> None:
        """Set baseline performance metric."""
        self.baseline_metric = metric
    
    def update(self, metric: float) -> bool:
        """
        Update with new metric and check for degradation.
        Returns True if retraining is recommended.
        """
        self.metric_history.append(metric)
        
        if len(self.metric_history) < self.window_size:
            return False
        
        recent = np.mean(self.metric_history[-self.window_size:])
        degradation = (self.baseline_metric - recent) / self.baseline_metric
        
        if degradation > self.metric_threshold:
            return True
        
        return False
