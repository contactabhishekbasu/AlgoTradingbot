# Automated Validation System

This directory contains the automated validation framework for AlgoTradingbot.

## Overview

The validation system provides comprehensive automated testing across 5 phases:

1. **System Setup**: Python version, packages, Docker, environment variables
2. **Data Layer**: API connectivity, database, cache, data validation
3. **ML Models**: Feature engineering, LSTM, XGBoost, ensemble, persistence
4. **Code Quality**: Unit tests, integration tests, linting, security scans
5. **Integration**: Performance benchmarks, end-to-end tests, concurrent operations

## Quick Start

```bash
# From project root
cd /path/to/AlgoTradingbot
source venv/bin/activate

# Run full validation
python scripts/validate_all.py

# Quick mode (faster)
python scripts/validate_all.py --quick

# Run specific phases
python scripts/validate_all.py --phases 1,2,3
```

## Architecture

### Core Components

```
scripts/validation/
├── __init__.py              # Package exports
├── base.py                  # Base classes and utilities
├── phase1_system.py         # Phase 1: System Setup
├── phase2_data.py           # Phase 2: Data Layer
├── phase3_ml.py             # Phase 3: ML Models
├── phase4_quality.py        # Phase 4: Code Quality
├── phase5_integration.py    # Phase 5: Integration
└── report_generator.py      # Report generation
```

### Base Classes

**`ValidationResult`**
- Represents the result of a single validation check
- Contains: name, status, message, details, duration, timestamp
- Status: PASS, FAIL, SKIP, WARN

**`ValidationPhase`**
- Collection of validation results for a phase
- Tracks: phase number, name, results, timing
- Provides: pass_rate, duration, summary statistics

**`BaseValidator`**
- Abstract base class for all validators
- Provides: `run_check()`, `run_async_check()` methods
- Handles: error capture, timing, result aggregation

### Validators

Each validator inherits from `BaseValidator` and implements specific checks:

**Phase1SystemValidator**
- Python version check
- Virtual environment check
- Required packages check
- Directory structure check
- Docker availability
- Docker Compose services
- Environment variables

**Phase2DataValidator**
- Yahoo Finance API connectivity
- Historical data fetching
- Data validation (OHLCV)
- Technical indicators calculation
- Database connection and storage
- Redis cache operations

**Phase3MLValidator**
- Feature engineering (50+ features)
- LSTM model training (with quick mode)
- XGBoost model training
- Ensemble model creation
- Model save/load persistence

**Phase4QualityValidator**
- Unit tests (pytest)
- Integration tests (pytest)
- Code linting (flake8)
- Security scanning (bandit)
- Type checking (mypy, optional)

**Phase5IntegrationValidator**
- Database performance benchmarks
- Cache performance benchmarks
- End-to-end prediction pipeline
- Backtesting engine
- Concurrent operations

## Usage Examples

### Basic Usage

```bash
# Full validation
python scripts/validate_all.py

# With options
python scripts/validate_all.py --quick --output ./reports
```

### Selective Validation

```bash
# System and data only
python scripts/validate_all.py --phases 1,2

# Skip ML (fastest)
python scripts/validate_all.py --no-ml

# Skip tests
python scripts/validate_all.py --no-tests
```

### CI/CD Integration

```bash
# Run validation and check exit code
python scripts/validate_all.py --no-reports
exit_code=$?

# Parse JSON report
if [ -f validation_reports/latest.json ]; then
    python -c "import json; print(json.load(open('validation_reports/latest.json'))['summary'])"
fi

exit $exit_code
```

## Report Formats

The system generates three report formats:

### HTML Report
- Beautiful, interactive web interface
- Color-coded status indicators
- Detailed results with timing
- Progress bars and statistics
- Open with: `open validation_reports/validation_report_*.html`

### JSON Report
- Machine-readable format
- Complete data structure
- Ideal for CI/CD integration
- Programmatic analysis

### Text Report
- Plain text summary
- Terminal-friendly
- Quick review
- Easy to parse

## Configuration

### Quick Mode

Quick mode reduces ML training epochs for faster validation:
- LSTM: 5 epochs instead of 10
- Accuracy threshold: 0.50 instead of 0.55
- Faster but less accurate

```bash
python scripts/validate_all.py --quick
```

### Custom Output Directory

```bash
python scripts/validate_all.py --output ~/Desktop/validation_reports
```

### Skip Report Generation

```bash
python scripts/validate_all.py --no-reports
```

## Extending the System

### Adding a New Check

1. Choose the appropriate phase validator (e.g., `phase2_data.py`)

2. Add a new check method:
```python
async def _check_new_feature(self):
    """Check new feature"""
    async def check():
        try:
            # Your validation logic here
            result = await some_validation()

            if result_is_valid:
                return True, "Success message", {'data': result}
            else:
                return False, "Failure message", {'error': 'details'}

        except Exception as e:
            return False, f"Failed: {str(e)}", {}

    result = await self.run_async_check("New Feature Check", check)
    print(f"  {result.status.value} New Feature: {result.message}")
```

3. Call it from `validate()` method:
```python
async def _run_async_checks(self):
    await self._check_existing_feature()
    await self._check_new_feature()  # Add here
```

### Adding a New Phase

1. Create `scripts/validation/phase6_newphase.py`:
```python
from .base import BaseValidator

class Phase6NewPhaseValidator(BaseValidator):
    def __init__(self):
        super().__init__(6, "New Phase Name")

    def validate(self):
        self.start()
        print(f"\n🔍 Phase {self.phase.phase_number}: {self.phase.phase_name}")
        # Add checks here
        self.end()
        return self.phase
```

2. Update `scripts/validation/__init__.py`:
```python
from .phase6_newphase import Phase6NewPhaseValidator

__all__ = [..., 'Phase6NewPhaseValidator']
```

3. Update `scripts/validate_all.py`:
```python
# Phase 6: New Phase
if 6 in selected_phases:
    validator = Phase6NewPhaseValidator()
    phase = validator.validate()
    completed_phases.append(phase)
    print_phase_summary(phase)
```

## Performance Considerations

### Memory Usage
- Phase 3 (ML Models) is most memory-intensive
- Use `--quick` mode to reduce memory usage
- Close other applications if needed
- Consider running phases separately

### Execution Time
- Full validation: ~45-60 minutes
- Quick mode: ~30-45 minutes
- Without ML: ~10-15 minutes
- Individual phases: 1-10 minutes each

### Network Usage
- Phase 2 fetches market data from Yahoo Finance
- Phase 5 may fetch additional data for integration tests
- Ensure stable internet connection
- Built-in retry logic for transient failures

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure you're in project root
cd /path/to/AlgoTradingbot

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Docker Service Errors**
```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs postgres
docker-compose logs redis
```

**Test Failures**
```bash
# Run health check first
python scripts/health_check.py

# Run specific phase
python scripts/validate_all.py --phases 1

# Check test output for details
pytest tests/unit/ -v
```

**Out of Memory**
```bash
# Use quick mode
python scripts/validate_all.py --quick

# Or skip ML
python scripts/validate_all.py --no-ml

# Or run phases separately
python scripts/validate_all.py --phases 1
python scripts/validate_all.py --phases 2
```

## Best Practices

1. **Run validation regularly**: After major changes, before deployment
2. **Use quick mode for iteration**: Faster feedback during development
3. **Review HTML reports**: Better visualization of issues
4. **Integrate with CI/CD**: Automate validation in your pipeline
5. **Track metrics over time**: Compare pass rates and performance
6. **Fix failures promptly**: Don't let technical debt accumulate

## Development

### Running Tests for Validators

```bash
# Test validation system itself
pytest tests/test_validation.py -v

# Test specific validator
pytest tests/test_validation.py::TestPhase1Validator -v
```

### Debugging

```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with Python debugger
python -m pdb scripts/validate_all.py
```

### Code Style

```bash
# Format code
black scripts/validation/

# Check linting
flake8 scripts/validation/ --max-line-length=120
```

## References

- [VALIDATION_GUIDE.md](../../VALIDATION_GUIDE.md) - Complete validation guide
- [VALIDATION_CHECKLIST.md](../../VALIDATION_CHECKLIST.md) - Quick reference checklist
- [Base Validators](base.py) - Core validation framework
- [Report Generator](report_generator.py) - Report generation system

## Version History

- **v2.0** (2025-10-31): Added automated validation system
- **v1.0** (2025-10-31): Initial manual validation documentation

## License

Part of the AlgoTradingbot project.
