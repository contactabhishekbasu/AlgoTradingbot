"""
Phase 4: Code Quality Validation
"""

import sys
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from .base import BaseValidator


class Phase4QualityValidator(BaseValidator):
    """Validator for code quality checks"""

    def __init__(self):
        super().__init__(4, "Code Quality")
        self.project_root = Path(__file__).parent.parent.parent

    def validate(self):
        """Run all Phase 4 validations"""
        self.start()

        print(f"\n🔍 Phase 4: {self.phase.phase_name}")
        print("=" * 70)

        # Run all checks
        self._check_unit_tests()
        self._check_integration_tests()
        self._check_linting()
        self._check_security_scan()
        self._check_type_checking()

        self.end()
        return self.phase

    def _check_unit_tests(self):
        """Run pytest unit tests"""
        def check():
            try:
                print("    Running unit tests (this may take a minute)...")

                result = subprocess.run(
                    ['pytest', 'tests/unit/', '-v', '--tb=short', '-q'],
                    capture_output=True,
                    text=True,
                    timeout=180,  # 3 minutes
                    cwd=self.project_root
                )

                output = result.stdout + result.stderr

                # Parse pytest output
                if 'passed' in output.lower():
                    # Extract pass count
                    import re
                    match = re.search(r'(\d+) passed', output)
                    passed = int(match.group(1)) if match else 0

                    match_failed = re.search(r'(\d+) failed', output)
                    failed = int(match_failed.group(1)) if match_failed else 0

                    if failed == 0 and passed > 0:
                        return True, f"{passed} tests passed", {
                            'passed': passed,
                            'failed': failed
                        }
                    else:
                        return False, f"{passed} passed, {failed} failed", {
                            'passed': passed,
                            'failed': failed
                        }
                else:
                    return False, "No tests found or failed to run", {}

            except subprocess.TimeoutExpired:
                return False, "Tests timed out after 3 minutes", {}
            except FileNotFoundError:
                return False, "pytest not found (install with: pip install pytest)", {}
            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = self.run_check("Unit Tests", check)
        print(f"  {result.status.value} Unit Tests: {result.message}")

    def _check_integration_tests(self):
        """Run pytest integration tests"""
        def check():
            try:
                print("    Running integration tests...")

                result = subprocess.run(
                    ['pytest', 'tests/integration/', '-v', '--tb=short', '-q', '-m', 'not slow'],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes
                    cwd=self.project_root
                )

                output = result.stdout + result.stderr

                # Parse pytest output
                if 'passed' in output.lower():
                    import re
                    match = re.search(r'(\d+) passed', output)
                    passed = int(match.group(1)) if match else 0

                    match_failed = re.search(r'(\d+) failed', output)
                    failed = int(match_failed.group(1)) if match_failed else 0

                    if failed == 0 and passed > 0:
                        return True, f"{passed} tests passed", {
                            'passed': passed,
                            'failed': failed
                        }
                    else:
                        return False, f"{passed} passed, {failed} failed", {
                            'passed': passed,
                            'failed': failed
                        }
                elif 'no tests ran' in output.lower() or result.returncode == 5:
                    # pytest exit code 5 means no tests collected
                    return True, "No integration tests found (skipping)", {
                        'passed': 0,
                        'skipped': True
                    }
                else:
                    return False, "Tests failed to run", {}

            except subprocess.TimeoutExpired:
                return False, "Tests timed out after 5 minutes", {}
            except FileNotFoundError:
                return False, "pytest not found", {}
            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = self.run_check("Integration Tests", check)
        print(f"  {result.status.value} Integration Tests: {result.message}")

    def _check_linting(self):
        """Run flake8 linting"""
        def check():
            try:
                result = subprocess.run(
                    ['flake8', 'src/', '--max-line-length=120', '--max-complexity=10',
                     '--count', '--statistics', '--exclude=__pycache__,*.pyc'],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=self.project_root
                )

                output = result.stdout + result.stderr

                # flake8 returns 0 if no issues, 1 if issues found
                if result.returncode == 0:
                    return True, "No linting errors", {'errors': 0}
                else:
                    # Count errors
                    import re
                    match = re.search(r'(\d+)\s+$', output.strip(), re.MULTILINE)
                    error_count = int(match.group(1)) if match else 'unknown'

                    return False, f"{error_count} linting errors found", {
                        'errors': error_count
                    }

            except FileNotFoundError:
                return False, "flake8 not found (install with: pip install flake8)", {}
            except subprocess.TimeoutExpired:
                return False, "Linting timed out", {}
            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = self.run_check("Code Linting", check)
        print(f"  {result.status.value} Code Linting: {result.message}")

    def _check_security_scan(self):
        """Run bandit security scan"""
        def check():
            try:
                result = subprocess.run(
                    ['bandit', '-r', 'src/', '-ll', '-f', 'json'],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=self.project_root
                )

                output = result.stdout

                # Parse JSON output
                try:
                    import json
                    data = json.loads(output)

                    high_issues = len([r for r in data.get('results', [])
                                     if r.get('issue_severity') == 'HIGH'])
                    medium_issues = len([r for r in data.get('results', [])
                                       if r.get('issue_severity') == 'MEDIUM'])
                    low_issues = len([r for r in data.get('results', [])
                                    if r.get('issue_severity') == 'LOW'])

                    if high_issues == 0 and medium_issues == 0:
                        return True, f"No high/medium issues ({low_issues} low)", {
                            'high': high_issues,
                            'medium': medium_issues,
                            'low': low_issues
                        }
                    else:
                        return False, f"{high_issues} high, {medium_issues} medium issues", {
                            'high': high_issues,
                            'medium': medium_issues,
                            'low': low_issues
                        }

                except json.JSONDecodeError:
                    # If JSON parsing fails, just check return code
                    if result.returncode == 0:
                        return True, "No security issues found", {}
                    else:
                        return False, "Security issues found", {}

            except FileNotFoundError:
                return False, "bandit not found (install with: pip install bandit)", {}
            except subprocess.TimeoutExpired:
                return False, "Security scan timed out", {}
            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = self.run_check("Security Scan", check)
        print(f"  {result.status.value} Security Scan: {result.message}")

    def _check_type_checking(self):
        """Run mypy type checking"""
        def check():
            try:
                result = subprocess.run(
                    ['mypy', 'src/', '--ignore-missing-imports', '--check-untyped-defs'],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=self.project_root
                )

                output = result.stdout + result.stderr

                # mypy returns 0 if no errors
                if result.returncode == 0 or 'Success' in output:
                    return True, "No type errors", {'errors': 0}
                else:
                    # Count errors
                    import re
                    matches = re.findall(r'error:', output)
                    error_count = len(matches)

                    return False, f"{error_count} type errors", {
                        'errors': error_count
                    }

            except FileNotFoundError:
                # mypy is optional
                return True, "mypy not installed (skipping)", {'skipped': True}
            except subprocess.TimeoutExpired:
                return False, "Type checking timed out", {}
            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = self.run_check("Type Checking", check)
        print(f"  {result.status.value} Type Checking: {result.message}")
