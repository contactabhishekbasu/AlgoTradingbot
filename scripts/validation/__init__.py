"""
Automated Validation Framework for AlgoTradingbot
"""

from .base import ValidationResult, ValidationPhase
from .phase1_system import Phase1SystemValidator
from .phase2_data import Phase2DataValidator
from .phase3_ml import Phase3MLValidator
from .phase4_quality import Phase4QualityValidator
from .phase5_integration import Phase5IntegrationValidator
from .report_generator import ReportGenerator

__all__ = [
    'ValidationResult',
    'ValidationPhase',
    'Phase1SystemValidator',
    'Phase2DataValidator',
    'Phase3MLValidator',
    'Phase4QualityValidator',
    'Phase5IntegrationValidator',
    'ReportGenerator',
]
