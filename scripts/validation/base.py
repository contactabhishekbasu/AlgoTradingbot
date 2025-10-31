"""
Base classes for validation framework
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import time


class ValidationStatus(Enum):
    """Status of a validation check"""
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    SKIP = "⊘ SKIP"
    WARN = "⚠️  WARN"


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    name: str
    status: ValidationStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def passed(self) -> bool:
        """Check if validation passed"""
        return self.status == ValidationStatus.PASS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'status': self.status.name,
            'status_display': self.status.value,
            'message': self.message,
            'details': self.details,
            'duration_ms': round(self.duration_ms, 2),
            'timestamp': self.timestamp.isoformat(),
            'passed': self.passed
        }


@dataclass
class ValidationPhase:
    """Results of a validation phase"""
    phase_number: int
    phase_name: str
    results: List[ValidationResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        """Get phase duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def passed_count(self) -> int:
        """Count of passed validations"""
        return sum(1 for r in self.results if r.passed)

    @property
    def total_count(self) -> int:
        """Total number of validations"""
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        """Pass rate as percentage"""
        if self.total_count == 0:
            return 0.0
        return (self.passed_count / self.total_count) * 100

    @property
    def passed(self) -> bool:
        """Check if all validations in phase passed"""
        return all(r.passed for r in self.results)

    def add_result(self, result: ValidationResult):
        """Add a validation result"""
        self.results.append(result)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'phase_number': self.phase_number,
            'phase_name': self.phase_name,
            'results': [r.to_dict() for r in self.results],
            'passed_count': self.passed_count,
            'total_count': self.total_count,
            'pass_rate': round(self.pass_rate, 2),
            'duration_seconds': round(self.duration_seconds, 2),
            'passed': self.passed,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
        }


class BaseValidator:
    """Base class for validators"""

    def __init__(self, phase_number: int, phase_name: str):
        self.phase = ValidationPhase(phase_number, phase_name)

    def start(self):
        """Mark phase as started"""
        self.phase.start_time = datetime.now()

    def end(self):
        """Mark phase as ended"""
        self.phase.end_time = datetime.now()

    def run_check(self, name: str, check_func, *args, **kwargs) -> ValidationResult:
        """
        Run a validation check and return result

        Args:
            name: Name of the check
            check_func: Function that returns (success: bool, message: str, details: dict)

        Returns:
            ValidationResult
        """
        start_time = time.time()

        try:
            success, message, details = check_func(*args, **kwargs)
            status = ValidationStatus.PASS if success else ValidationStatus.FAIL

            duration_ms = (time.time() - start_time) * 1000

            result = ValidationResult(
                name=name,
                status=status,
                message=message,
                details=details or {},
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                name=name,
                status=ValidationStatus.FAIL,
                message=f"Error: {str(e)}",
                details={'exception': str(e), 'type': type(e).__name__},
                duration_ms=duration_ms
            )

        self.phase.add_result(result)
        return result

    async def run_async_check(self, name: str, check_func, *args, **kwargs) -> ValidationResult:
        """
        Run an async validation check and return result

        Args:
            name: Name of the check
            check_func: Async function that returns (success: bool, message: str, details: dict)

        Returns:
            ValidationResult
        """
        start_time = time.time()

        try:
            success, message, details = await check_func(*args, **kwargs)
            status = ValidationStatus.PASS if success else ValidationStatus.FAIL

            duration_ms = (time.time() - start_time) * 1000

            result = ValidationResult(
                name=name,
                status=status,
                message=message,
                details=details or {},
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                name=name,
                status=ValidationStatus.FAIL,
                message=f"Error: {str(e)}",
                details={'exception': str(e), 'type': type(e).__name__},
                duration_ms=duration_ms
            )

        self.phase.add_result(result)
        return result

    def validate(self) -> ValidationPhase:
        """
        Run all validations for this phase
        Must be implemented by subclasses

        Returns:
            ValidationPhase with results
        """
        raise NotImplementedError("Subclasses must implement validate()")
