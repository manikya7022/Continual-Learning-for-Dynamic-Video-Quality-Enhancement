"""
Knowledge Distillation for Continual Learning.
Uses the previous model as teacher to prevent forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from copy import deepcopy


class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss combining task and distillation losses."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Video output - use MSE for distillation
        distill_loss = F.mse_loss(student_output, teacher_output.detach())
        
        if target is not None:
            task_loss = F.mse_loss(student_output, target)
            loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss
        else:
            loss = distill_loss
        
        return loss


class ContinualDistillation:
    """Continual learning wrapper using knowledge distillation."""
    
    def __init__(self, model: nn.Module, temperature: float = 4.0, alpha: float = 0.5):
        self.student = model
        self.teacher = None
        self.distill_loss = DistillationLoss(temperature, alpha)
        self.task_count = 0
    
    def register_task(self) -> None:
        """Create teacher from current student."""
        self.teacher = deepcopy(self.student)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.task_count += 1
    
    def compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, task_loss_fn: nn.Module
    ) -> Dict[str, torch.Tensor]:
        student_output = self.student(inputs)
        task_loss = task_loss_fn(student_output, targets)
        
        losses = {'task': task_loss, 'distill': torch.tensor(0.0), 'total': task_loss}
        
        if self.teacher is not None:
            with torch.no_grad():
                teacher_output = self.teacher(inputs)
            distill_loss = self.distill_loss(student_output, teacher_output, targets)
            losses['distill'] = distill_loss
            losses['total'] = losses['task'] + distill_loss
        
        return losses
