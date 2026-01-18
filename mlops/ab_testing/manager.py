"""
A/B Testing Framework for Model Deployment.
"""

import hashlib
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from scipy import stats


@dataclass
class Variant:
    """A/B test variant."""
    name: str
    model_version: str
    traffic_percentage: float = 50.0
    metrics: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result of A/B test analysis."""
    winner: Optional[str]
    is_significant: bool
    p_value: float
    effect_size: float
    confidence_interval: tuple


class ABTestManager:
    """
    Manage A/B tests for model deployment.
    
    Features:
        - Deterministic user assignment
        - Statistical significance testing
        - Automatic winner detection
    """
    
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}
        self.active_experiment: Optional[str] = None
    
    def create_experiment(
        self,
        name: str,
        control_model: str,
        treatment_model: str,
        control_percentage: float = 95.0,
    ) -> None:
        """Create new A/B test experiment."""
        self.experiments[name] = {
            'control': Variant(
                name='control',
                model_version=control_model,
                traffic_percentage=control_percentage,
            ),
            'treatment': Variant(
                name='treatment',
                model_version=treatment_model,
                traffic_percentage=100 - control_percentage,
            ),
            'start_time': datetime.now(),
            'status': 'running',
        }
        self.active_experiment = name
    
    def assign_variant(self, user_id: str, experiment: Optional[str] = None) -> str:
        """Deterministically assign user to variant."""
        exp_name = experiment or self.active_experiment
        if exp_name is None or exp_name not in self.experiments:
            return 'control'
        
        exp = self.experiments[exp_name]
        
        # Hash user_id for deterministic assignment
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        
        if hash_val < exp['control'].traffic_percentage:
            return 'control'
        return 'treatment'
    
    def record_metric(
        self,
        experiment: str,
        variant: str,
        metric_name: str,
        value: float,
    ) -> None:
        """Record metric observation."""
        if experiment not in self.experiments:
            return
        
        exp = self.experiments[experiment]
        v = exp[variant]
        
        if metric_name not in v.metrics:
            v.metrics[metric_name] = []
        v.metrics[metric_name].append(value)
    
    def analyze(
        self,
        experiment: str,
        metric_name: str,
        min_samples: int = 100,
    ) -> ExperimentResult:
        """Analyze experiment results."""
        if experiment not in self.experiments:
            raise ValueError(f"Experiment {experiment} not found")
        
        exp = self.experiments[experiment]
        control = exp['control'].metrics.get(metric_name, [])
        treatment = exp['treatment'].metrics.get(metric_name, [])
        
        if len(control) < min_samples or len(treatment) < min_samples:
            return ExperimentResult(
                winner=None,
                is_significant=False,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0, 0),
            )
        
        # T-test
        t_stat, p_value = stats.ttest_ind(treatment, control)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(control) + np.var(treatment)) / 2)
        effect_size = (np.mean(treatment) - np.mean(control)) / pooled_std
        
        # Confidence interval
        mean_diff = np.mean(treatment) - np.mean(control)
        se = np.sqrt(np.var(control)/len(control) + np.var(treatment)/len(treatment))
        ci = (mean_diff - 1.96*se, mean_diff + 1.96*se)
        
        is_significant = p_value < 0.05
        winner = None
        if is_significant:
            winner = 'treatment' if mean_diff > 0 else 'control'
        
        return ExperimentResult(
            winner=winner,
            is_significant=is_significant,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
        )
    
    def conclude_experiment(self, experiment: str, promote_winner: bool = True) -> str:
        """Conclude experiment and optionally promote winner."""
        result = self.analyze(experiment, 'vmaf')
        
        self.experiments[experiment]['status'] = 'concluded'
        self.experiments[experiment]['result'] = result
        
        if promote_winner and result.winner:
            return self.experiments[experiment][result.winner].model_version
        
        return self.experiments[experiment]['control'].model_version
