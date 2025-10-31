#!/usr/bin/env python3
"""
Automated Validation Script for AlgoTradingbot

This script automates the entire validation process across all 5 phases:
1. System Setup
2. Data Layer
3. ML Models
4. Code Quality
5. Integration & Performance

Usage:
    python scripts/validate_all.py                    # Run all phases
    python scripts/validate_all.py --phases 1,2,3     # Run specific phases
    python scripts/validate_all.py --quick            # Quick mode (faster ML training)
    python scripts/validate_all.py --no-ml            # Skip ML model training
    python scripts/validate_all.py --help             # Show help
"""

import sys
import argparse
import platform
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import validators
from scripts.validation.base import ValidationPhase
from scripts.validation.phase1_system import Phase1SystemValidator
from scripts.validation.phase2_data import Phase2DataValidator
from scripts.validation.phase3_ml import Phase3MLValidator
from scripts.validation.phase4_quality import Phase4QualityValidator
from scripts.validation.phase5_integration import Phase5IntegrationValidator
from scripts.validation.report_generator import ReportGenerator


class Colors:
    """Terminal colors for better output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_banner():
    """Print validation banner"""
    banner = f"""
{Colors.BLUE}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          AlgoTradingbot Automated Validation System                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
{Colors.END}
    """
    print(banner)


def print_phase_summary(phase: ValidationPhase):
    """Print summary for a phase"""
    status_symbol = f"{Colors.GREEN}✅{Colors.END}" if phase.passed else f"{Colors.RED}❌{Colors.END}"

    print(f"\n{Colors.BOLD}Phase {phase.phase_number} Summary:{Colors.END}")
    print(f"  Status: {status_symbol} {'PASS' if phase.passed else 'FAIL'}")
    print(f"  Checks: {phase.passed_count}/{phase.total_count} passed ({phase.pass_rate:.1f}%)")
    print(f"  Duration: {phase.duration_seconds:.2f}s")


def print_overall_summary(phases: list, total_duration: float):
    """Print overall validation summary"""
    total_checks = sum(p.total_count for p in phases)
    passed_checks = sum(p.passed_count for p in phases)
    all_passed = all(p.passed for p in phases)

    pass_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0

    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}Overall Validation Summary{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.END}")

    status_color = Colors.GREEN if all_passed else Colors.RED
    status_text = "✅ PASS" if all_passed else "❌ FAIL"

    print(f"\n{status_color}{Colors.BOLD}Status: {status_text}{Colors.END}")
    print(f"\nPhases: {len([p for p in phases if p.passed])}/{len(phases)} passed")
    print(f"Checks: {passed_checks}/{total_checks} passed ({pass_rate:.1f}%)")
    print(f"Total Duration: {total_duration:.2f}s ({total_duration/60:.1f} minutes)")

    # Print failed phases if any
    failed_phases = [p for p in phases if not p.passed]
    if failed_phases:
        print(f"\n{Colors.RED}{Colors.BOLD}Failed Phases:{Colors.END}")
        for phase in failed_phases:
            failed_checks = [r for r in phase.results if not r.passed]
            print(f"  • Phase {phase.phase_number}: {phase.phase_name} ({len(failed_checks)} failures)")
            for check in failed_checks:
                print(f"      - {check.name}: {check.message}")

    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.END}\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Automated validation system for AlgoTradingbot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_all.py                    # Run all phases
  python scripts/validate_all.py --phases 1,2       # Run phases 1 and 2 only
  python scripts/validate_all.py --quick            # Quick mode with reduced ML training
  python scripts/validate_all.py --no-ml            # Skip ML model training (phases 1,2,4,5)
  python scripts/validate_all.py --no-tests         # Skip code quality tests
  python scripts/validate_all.py --output ./reports # Custom output directory
        """
    )

    parser.add_argument(
        '--phases',
        type=str,
        help='Comma-separated list of phases to run (e.g., "1,2,3")'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: Use fewer epochs for ML training (faster but less accurate)'
    )

    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Skip ML model training (Phase 3)'
    )

    parser.add_argument(
        '--no-tests',
        action='store_true',
        help='Skip code quality tests (Phase 4)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='validation_reports',
        help='Output directory for validation reports (default: validation_reports)'
    )

    parser.add_argument(
        '--no-reports',
        action='store_true',
        help='Skip generating HTML/JSON reports'
    )

    return parser.parse_args()


def main():
    """Main validation function"""
    args = parse_arguments()

    print_banner()

    # Determine which phases to run
    if args.phases:
        selected_phases = [int(p.strip()) for p in args.phases.split(',')]
    elif args.no_ml and args.no_tests:
        selected_phases = [1, 2, 5]
    elif args.no_ml:
        selected_phases = [1, 2, 4, 5]
    elif args.no_tests:
        selected_phases = [1, 2, 3, 5]
    else:
        selected_phases = [1, 2, 3, 4, 5]

    print(f"{Colors.CYAN}Configuration:{Colors.END}")
    print(f"  Phases: {', '.join(map(str, selected_phases))}")
    print(f"  Quick Mode: {'Yes' if args.quick else 'No'}")
    print(f"  Output Directory: {args.output}")
    print()

    # Collect metadata
    metadata = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'quick_mode': args.quick,
        'phases_run': selected_phases,
    }

    # Run validations
    start_time = datetime.now()
    completed_phases = []

    try:
        # Phase 1: System Setup
        if 1 in selected_phases:
            validator = Phase1SystemValidator()
            phase = validator.validate()
            completed_phases.append(phase)
            print_phase_summary(phase)

        # Phase 2: Data Layer
        if 2 in selected_phases:
            validator = Phase2DataValidator()
            phase = validator.validate()
            completed_phases.append(phase)
            print_phase_summary(phase)

        # Phase 3: ML Models
        if 3 in selected_phases:
            validator = Phase3MLValidator(quick_mode=args.quick)
            phase = validator.validate()
            completed_phases.append(phase)
            print_phase_summary(phase)

        # Phase 4: Code Quality
        if 4 in selected_phases:
            validator = Phase4QualityValidator()
            phase = validator.validate()
            completed_phases.append(phase)
            print_phase_summary(phase)

        # Phase 5: Integration
        if 5 in selected_phases:
            validator = Phase5IntegrationValidator()
            phase = validator.validate()
            completed_phases.append(phase)
            print_phase_summary(phase)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Validation interrupted by user{Colors.END}")
        return 1
    except Exception as e:
        print(f"\n\n{Colors.RED}Validation failed with error: {str(e)}{Colors.END}")
        import traceback
        traceback.print_exc()
        return 1

    # Calculate total duration
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    # Print overall summary
    print_overall_summary(completed_phases, total_duration)

    # Generate reports
    if not args.no_reports:
        print(f"{Colors.CYAN}Generating reports...{Colors.END}")

        report_gen = ReportGenerator(output_dir=Path(args.output))
        report_paths = report_gen.generate_all_reports(completed_phases, metadata)

        print(f"\n{Colors.GREEN}Reports generated:{Colors.END}")
        print(f"  HTML: {report_paths['html']}")
        print(f"  JSON: {report_paths['json']}")
        print(f"  Text: {report_paths['text']}")
        print()

    # Return exit code
    all_passed = all(p.passed for p in completed_phases)
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
